import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import math
from typing import Any, Dict, Tuple, List, Union, Set
import warnings
from collections import OrderedDict
from torch.nn.modules.linear import Linear
from torchmeta.modules import MetaLinear, MetaSequential, MetaModule
from .meta_bert import MetaBertModel
from .basics import CNNEncoder
from tqdm import tqdm

class LabelMap(MetaModule):
    def __init__(self, hidden_dim:int, input_dim:int, levelwise:bool=True):
        super().__init__()
        self.model = CNNEncoder(input_size=input_dim, output_size=hidden_dim)
        self.levelwise = levelwise
        self.labels = None
        self.label_masks = None
        self.output_dim = hidden_dim * 3 if levelwise else hidden_dim

    def forward(self, embeddings=None, masks=None, params=None):
        if embeddings is None:
            embeddings = self.labels
            masks = self.label_masks
        if self.levelwise:
            _, output0 = self.model(embeddings[0, :], masks, params)
            _, output1 = self.model(embeddings[1, :], masks, params)
            _, output2 = self.model(embeddings[2, :], masks, params)
            return torch.cat((output0, output1, output2), dim=-1)
        else:
            _, output = self.model(embeddings[0, :], masks, params)
            return output

class InputMap(MetaModule):
    def __init__(self):
        super().__init__()
        self.model = MetaBertModel.from_pretrained("bert-large-cased")
        self.output_dim = 2048
    def forward(self, input_ids, attention_masks, spans, *args, **kwargs):
        encoded, _ = self.model(input_ids, attention_masks, *args, **kwargs)
        encoded = torch.gather(encoded, 1, spans.unsqueeze(-1).repeat(1, 1, encoded.size(2))).flatten(start_dim=1)
        return encoded

class ZIE(MetaModule):
    def __init__(self,hidden_dim:int,device:Union[torch.device, None]=None,**kwargs)->None:
        super().__init__()
        self.input_map = InputMap()
        # self.label_map = LabelMap(hidden_dim=hidden_dim, input_dim=1024)
        # self.map_mat = MetaLinear(in_features=self.input_map.output_dim, out_features=self.label_map.output_dim, bias=False)
        self.map_mat = MetaLinear(in_features=self.input_map.output_dim, out_features=1024, bias=False)
        self.crit = nn.CrossEntropyLoss()
        self.gamma = 0.2
        self.device = device
        self.to(device=device)
        self.maml = True
        self.outputs = {}
        self.history = None
        self.labels = None
        self.label_attention_masks = None

    def set(self, features:torch.tensor, ids:Union[int, torch.Tensor, List, None]=None, max_id:int=-1):
        with torch.no_grad():
            if isinstance(ids, (torch.Tensor, list)):
                if torch.any(ids > self.nslots):
                    warnings.warn("Setting features to new classes. Using 'extend' or 'append' is preferred for new classes")
                self.classes.weight[ids] = features
            elif isinstance(ids, int):
                self.classes.weight[ids] = features
            else:
                if max_id == -1:
                    raise ValueError(f"Need input for either ids or max_id")
                self.classes.weight[:max_id] = features

    def append(self, feature):
        with torch.no_grad():
            self.classes.weight[self.nslots] = feature
            self.nslots += 1

    def extend(self, features):
        with torch.no_grad():
            features = features.to(self.device)
            if len(features.size()) == 1:
                warnings.warn("Extending 1-dim feature vector. Using 'append' instead is preferred.")
                self.append(features)
            else:
                nclasses = features.size(0)
                self.classes.weight[self.nslots:self.nslots+nclasses] = features
                self.nslots += nclasses

    @property
    def mask(self,):
        self._mask[:, :self.nslots] = 0
        self._mask[:, self.nslots:] = float("-inf")
        return self._mask

    def idx_mask(self, idx:Union[torch.LongTensor, int, List[int], None]=None, max_idx:Union[torch.LongTensor, int, None]=None):
        assert (idx is not None) or (max_idx is not None)
        assert (idx is None) or (max_idx is None)
        mask = torch.zeros_like(self._mask) + float("-inf")
        if idx is not None:
            mask[:, idx] = 0
        if max_idx is not None:
            if isinstance(max_idx, torch.LongTensor):
                max_idx = max_idx.item()
            mask[:, :max_idx] = 0
        return mask

    @property
    def features(self):
        return self.classes.weight[:self.nslots]

    def set_labels(self, labels:torch.LongTensor, label_masks:Union[None, torch.BoolTensor]=None):
        # labels = F.embedding(labels, self.input_map.model.meta_named_parameters['embeddings.word_embeddings.weight'], padding_idx = 0)
        # self.label_map.labels = labels.detach().to(self.device)
        # if label_masks is None:
        #     label_masks = labels > 0
        # else:
        #     self.label_map.label_masks = label_masks.detach().to(self.device)
        self.labels = labels.detach().to(self.device)
        self.label_attention_masks = label_masks.detach().to(self.device)

    def forward(self, batch, predict:bool=False, batch_neg:bool=False, zcenter=False, return_loss:bool=True, return_feature:bool=False, log_outputs:bool=True, params=None):
        inputs, attention_masks, label_mat = batch.token_ids, batch.attention_masks, batch.label_mat[:, 1:]
        labels = getattr(batch, "labels", None)
        label_attention_masks = getattr(batch, "label_attention_masks", None)
        encoded = self.input_map(inputs, attention_masks, batch.spans, params=self.get_subdict(params, "input_map"))
        encoded = self.map_mat(encoded)

        _, labels = self.input_map.model(self.labels, self.label_attention_masks)

        scores = torch.matmul(encoded, labels.transpose(0, 1))
        valid = label_mat != 0
        inval = label_mat == 0
        loss_valid = torch.sum(torch.relu(0.5 * self.gamma - scores[valid]))
        loss_inval = torch.sum(torch.relu(scores[inval] + 0.5 * self.gamma))
        loss = loss_valid + loss_inval
        if log_outputs:
            raw_pred = scores > 0
            acc = torch.mean((raw_pred.long() == label_mat.long()).float())
            idx, num = torch.nonzero(raw_pred, as_tuple=True)
            print(idx)
            label_pred = torch.zeros(raw_pred.size(0), device=self.device).to(num)
            label_pred[idx] = num + 1
            self.outputs["accuracy"] = acc.item()
            self.outputs["prediction"] = label_pred.detach().cpu()
            self.outputs["label"] = torch.nonzero(batch.label_mat, as_tuple=True)[1].detach().cpu()
            self.outputs["input_features"] = inputs.detach().cpu()
            self.outputs["encoded_features"] = encoded.detach().cpu()
        if return_loss:
            return loss
        else:
            return scores

    def score(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def clone_params(self,):
        return OrderedDict({k:v.clone().detach() for k,v in self.meta_named_parameters()})

    def set_history(self,):
        self.history = {"params": self.clone_params(), "nslots": self.nslots}

    def set_exemplar(self, dataloader, q:int=20, params=None, label_sets:Union[List, Set, None]=None, collect_none:bool=False, use_input:bool=False, output_only:bool=False, output:Union[str, None]=None):
        self.eval()
        with torch.no_grad():
            ifeat = []; ofeat = []; label = []
            num_batches = len(dataloader)
            for batch in tqdm(dataloader, "collecting exemplar", ncols=128):
                batch = batch.to(self.device)
                loss = self.forward(batch, params=params)
                ifeat.append(self.outputs["input_features"])
                if use_input:
                    ofeat.append(self.outputs["input_features"])
                else:
                    ofeat.append(self.outputs["encoded_features"])
                label.append(self.outputs["label"])
            ifeat = torch.cat(ifeat, dim=0)
            ofeat = torch.cat(ofeat, dim=0)
            label = torch.cat(label, dim=0)
            nslots = max(self.nslots, torch.max(label).item()+1)
            exemplar = {}
            if label_sets is None:
                if collect_none:
                    label_sets = range(nslots)
                else:
                    label_sets = range(1, nslots)
            else:
                if collect_none:
                    if 0 not in label_sets:
                        label_sets = sorted([0] + list(label_sets))
                    else:
                        label_sets = sorted(list(label_sets))
                else:
                    label_sets = sorted([t for t in label_sets if t != 0])
            for i in label_sets:
                idx = (label == i)
                if i == 0:
                    # random sample for none type
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    exemplar[i] = numpy.random.choice(nidx, q, replace=False).tolist()
                    continue
                if torch.any(idx):
                    exemplar[i] = []
                    nidx = torch.nonzero(idx, as_tuple=True)[0].tolist()
                    mfeat = torch.mean(ofeat[idx], dim=0, keepdims=True)
                    if len(nidx) < q:
                        exemplar[i].extend(nidx * (q // len(nidx)) + nidx[:(q % len(nidx))])
                    else:
                        for j in range(q):
                            if j == 0:
                                dfeat = torch.sum((ofeat[nidx] - mfeat)**2, dim=1)
                            else:
                                cfeat = ofeat[exemplar[i]].sum(dim=0, keepdims=True)
                                cnum = len(exemplar[i])
                                dfeat = torch.sum((mfeat * (cnum + 1) - ofeat[nidx] - cfeat)**2, )
                            tfeat = torch.argmin(dfeat)
                            exemplar[i].append(nidx[tfeat])
                            nidx.pop(tfeat.item())
            exemplar = {i: ifeat[v] for i,v in exemplar.items()}
            exemplar_features = []
            exemplar_labels = []
            for label, features in exemplar.items():
                exemplar_features.append(features)
                exemplar_labels.extend([label]*features.size(0))
            exemplar_features = torch.cat(exemplar_features, dim=0).cpu()
            exemplar_labels = torch.LongTensor(exemplar_labels).cpu()
            if not output_only or output is not None:
                if output == "train" or output is None:
                    if self.exemplar_features is None:
                        self.exemplar_features = exemplar_features
                        self.exemplar_labels = exemplar_labels
                    else:
                        self.exemplar_features = torch.cat((self.exemplar_features, exemplar_features), dim=0)
                        self.exemplar_labels = torch.cat((self.exemplar_labels, exemplar_labels), dim=0)
                elif output == "dev":
                    if self.dev_exemplar_features is None:
                        self.dev_exemplar_features = exemplar_features
                        self.dev_exemplar_labels = exemplar_labels
                    else:
                        self.dev_exemplar_features = torch.cat((self.dev_exemplar_features, exemplar_features), dim=0)
                        self.dev_exemplar_labels = torch.cat((self.dev_exemplar_labels, exemplar_labels), dim=0)

        return {i: v.cpu() for i,v in exemplar.items()}

    def initialize(self, exemplar, ninstances:Dict[int, int], gamma:float=1.0, tau:float=1.0, alpha:float=0.5, params=None):
        self.eval()

        with torch.no_grad():
            weight_norm = torch.norm(self.classes.weight[1:self.nslots], dim=1).mean(dim=0)
            label_inits = []
            label_kt = {}
            for label, feats in exemplar.items():
                exemplar_inputs = self.input_map(feats.to(self.device), params=self.get_subdict(params, "input_map"))
                exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                exemplar_scores = exemplar_scores + self.mask
                exemplar_scores[:, 0] = 0
                exemplar_weights = torch.softmax(exemplar_scores * tau, dim=1)
                normalized_inputs = exemplar_inputs / torch.norm(exemplar_inputs, dim=1, keepdim=True) * weight_norm
                proto = (exemplar_weights[:, :1] * normalized_inputs).mean(dim=0)
                knowledge = torch.matmul(exemplar_weights[:, 1:self.nslots], self.classes.weight[1:self.nslots]).mean(dim=0)
                gate = alpha * math.exp(- ninstances[label] * gamma)
                # gate = 1 / (1 + ninstances[label] * gamma)
                rnd = torch.randn_like(proto) * weight_norm / math.sqrt(self.classes.weight.size(1))
                initvec = proto *  gate + knowledge * gate + (1 - gate) * rnd
                label_inits.append((label, initvec.cpu()))
                label_kt[label] = exemplar_weights.mean(dim=0).cpu()
            label_inits.sort(key=lambda t:t[0])
            inits = []
            for i, (label, init) in enumerate(label_inits):
                assert label == self.nslots + i
                inits.append(init)
            inits = torch.stack(inits, dim=0)
            self.outputs["new2old"] = label_kt
        return inits.detach()

    def initialize2(self, exemplar, ninstances:Dict[int, int], gamma:float=1.0, tau:float=1.0, alpha:float=0.5, delta:float=0.5, params=None):
        self.eval()
        def top_p(probs, p=0.9):
            _val, _idx = torch.sort(probs, descending=True, dim=1)
            top_mask = torch.zeros_like(probs).float() - float("inf")
            for _type in range(probs.size(0)):
                accumulated = 0
                _n = 0
                while accumulated < p or _n <= 1:
                    top_mask[_type, _idx[_type, _n]] = 0
                    accumulated += _val[_type, _n]
                    _n += 1
            return top_mask
        with torch.no_grad():
            weight_norm = torch.norm(self.classes.weight[1:self.nslots], dim=1).mean(dim=0)
            label_inits = []
            label_kt = {}
            for label, feats in exemplar.items():
                exemplar_inputs = self.input_map(feats.to(self.device), params=self.get_subdict(params, "input_map"))
                exemplar_scores = self.classes(exemplar_inputs, params=self.get_subdict(params, "classes"))
                exemplar_scores = exemplar_scores + self.mask
                exemplar_scores[:, 0] = 0
                top_mask = top_p(torch.softmax(exemplar_scores, dim=1))
                exemplar_scores = exemplar_scores + top_mask
                exemplar_scores[:, 0] = 0
                exemplar_weights = torch.softmax(exemplar_scores * tau, dim=1)
                normalized_inputs = exemplar_inputs / torch.norm(exemplar_inputs, dim=1, keepdim=True) * weight_norm
                proto = delta * (exemplar_weights[:, :1] * normalized_inputs).mean(dim=0)
                kweight = (1 - exemplar_weights[:, :1])
                knowledge = torch.matmul((1-delta*exemplar_weights[:, :1]) * (exemplar_weights[:, 1:self.nslots] + 1e-8) / torch.clamp(1 - exemplar_weights[:, :1], 1e-8), self.classes.weight[1:self.nslots]).mean(dim=0)
                gate = alpha * math.exp(- ninstances[label] * gamma)
                rnd = torch.randn_like(proto) * weight_norm / math.sqrt(self.classes.weight.size(1))
                initvec = proto *  gate + knowledge * gate + (1 - gate) * rnd
                if torch.any(torch.isnan(initvec)):
                    print(proto, knowledge, rnd, gate, exemplar_weights[:, :1], exemplar_scores[-1, :self.nslots])
                    input()
                label_inits.append((label, initvec.cpu()))
                label_kt[label] = exemplar_weights.mean(dim=0).cpu()
            label_inits.sort(key=lambda t:t[0])
            inits = []
            for i, (label, init) in enumerate(label_inits):
                assert label == self.nslots + i
                inits.append(init)
            inits = torch.stack(inits, dim=0)
            self.outputs["new2old"] = label_kt
        return inits.detach()


def test(): # sanity check
    m = ZIE(hidden_dim=512, device=torch.device("cpu"))

if __name__ == "__main__":
    test()
