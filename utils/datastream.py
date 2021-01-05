from typing import Iterable, List, Tuple, Dict, Union
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader#, Sampler
import json
import os
from transformers import BertTokenizerFast
from tqdm import tqdm
class Instance(object):
    '''
    - piece_ids: L
    - label: 1
    - span: 2
    - feature_path: str
    - sentence_id: str
    - mention_id: str
    '''
    def __init__(self, token_ids:List[int], label:List[int], span:Tuple[int, int], sentence_id:str, mention_id:str) -> None:
        self.token_ids = token_ids
        self.label = label
        self.span = span
        # self.features = features
        self.sentence_id = sentence_id
        self.mention_id = mention_id

    def totensor(self,):
        if not isinstance(self.token_ids, torch.LongTensor):
            self.token_ids = torch.LongTensor(self.token_ids)
        if not isinstance(self.span, torch.LongTensor):
            self.span = torch.LongTensor(self.span)
        if not isinstance(self.label, torch.LongTensor):
            self.label = torch.LongTensor(self.label)
        # if not isinstance(self.features, torch.FloatTensor):
        #     self.features = torch.FloatTensor(self.features)
        return self

    def todict(self,):
        return {
            "token_ids": self.token_ids,
            "span": self.span,
            "label": self.label,
            "sentence_id": self.sentence_id,
            "mention_id": self.mention_id
        }

    def load_clone(self,):
        # if isinstance(self.features, str):
        #     if not self.features.endswith("npy"):
        #         self.features += ".npy"
        #     npy_features = np.load(self.features)
        #     npy_features = npy_features[self.span, :]
        #     features = torch.from_numpy(npy_features).float().flatten()
        # else:
        #     features = self.features
        return self.__class__(
            token_ids=self.token_ids,
            label=self.label,
            span=self.span,
            # features=features,
            sentence_id=self.sentence_id,
            mention_id=self.mention_id
        )

class ZeroShotDataCollection(object):
    dataset = "ace"

    def _transform_oneie(self, oneie, tokenizer, ontology=None):
        instances = []
        tokens = None; start = None; end = None; mid = None; label = None
        if ontology is None:
            ontology = {"NA": 0}
        for sent in tqdm(oneie):
            tokens = sent["pieces"]
            _token_lens = sent["token_lens"]
            sent_id = sent["sent_id"]
            positive_spans = set()
            for entity in sent["entity_mentions"]:
                mid = entity["id"]
                start = sum(_token_lens[:entity["start"]])
                end = start - 1 + sum(_token_lens[entity["start"]:entity["end"]])
                label_name = entity["entity_type"]
                if label_name not in ontology:
                    ontology[label_name] = len(ontology)
                label = ontology[label_name]
                instances.append((tokens, start, end, label, mid, sent_id))
                positive_spans.add((start, end))

            nneg = 0
            for nstart in range(len(tokens)):
                if (nstart, nstart) not in positive_spans:
                    mid = f"{sent_id}-NA-{nneg}"
                    nneg += 1
                    instances.append((tokens, nstart, nstart, 0, mid, sent_id))
        instance_list = []
        for entry in instances:
            label_vec = [0] * len(ontology)
            label_vec[entry[3]] = 1
            token_ids = tokenizer.encode(
                entry[0],
                add_special_tokens=True,
                max_length=self.max_length,
                is_pretokenized=True,
                truncation=True
            )
            instance_list.append(
                Instance(
                    token_ids=token_ids,
                    label=label_vec,
                    span=(entry[1], entry[2]),
                    sentence_id=entry[5],
                    mention_id=entry[4]
                )
            )
        return instance_list, ontology

    def __init__(self, root:str) -> None:
        self.max_length = 512
        self.__preprocess(root)

    def __preprocess(self, root:str) -> None:
        self.root = root
        train_file = os.path.join(root, self.dataset, f"{self.dataset}.train.processed.json")
        train_ontology_file = os.path.join(root, self.dataset, f"{self.dataset}.train_ontology.processed.json")
        dev_file = os.path.join(root, self.dataset, f"{self.dataset}.dev.processed.json")
        dev_ontology_file = os.path.join(root, self.dataset, f"{self.dataset}.dev_ontology.processed.json")
        test_file = os.path.join(root, self.dataset, f"{self.dataset}.test.processed.json")
        test_ontology_file = os.path.join(root, self.dataset, f"{self.dataset}.test_ontology.processed.json")
        if os.path.exists(train_file) and os.path.exists(train_ontology_file) and \
            os.path.exists(dev_file) and os.path.exists(dev_ontology_file) and \
                os.path.exists(test_file) and os.path.exists(test_ontology_file):
            train = json.load(open(train_file))
            self.train = [Instance(**t) for t in train]
            self.train_ontology = json.load(open(train_ontology_file))
            dev = json.load(open(dev_file))
            self.dev = [Instance(**t) for t in dev]
            self.dev_ontology = json.load(open(dev_ontology_file))
            test = json.load(open(test_file))
            self.test = [Instance(**t) for t in test]
            self.test_ontology = json.load(open(test_ontology_file))
            return


        tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
        def collect_dataset_split(dataset, split):
            json_f = os.path.join(root, dataset, f"{dataset}.{split}.json")
            jsonl_f = os.path.join(root, dataset, f"{dataset}.{split}.jsonl")
            if os.path.exists(json_f):
                with open(json_f, "rt") as fp:
                    d = json.load(fp)
            else:
                with open(jsonl_f, "rt") as fp:
                    d = [json.loads(t) for t in fp]
            return d

        train_split = collect_dataset_split(self.dataset, "train")
        train_instances, train_ontology = self._transform_oneie(train_split, tokenizer)
        self.train = train_instances
        self.train_ontology = train_ontology
        json.dump(self.train_ontology, open(train_ontology_file, "wt"), indent=4)
        json.dump([t.todict() for t in self.train], open(train_file, "wt"), indent=4)

        dev_split = collect_dataset_split(self.dataset, "dev")
        dev_instances, dev_ontology = self._transform_oneie(dev_split, tokenizer, train_ontology.copy())
        self.dev = dev_instances
        self.dev_ontology = dev_ontology
        json.dump(self.dev_ontology, open(dev_ontology_file, "wt"), indent=4)
        json.dump([t.todict() for t in self.dev], open(dev_file, "wt"), indent=4)


        test_split = collect_dataset_split(self.dataset, "test")
        test_instances, test_ontology = self._transform_oneie(test_split, tokenizer, train_ontology.copy())
        self.test = test_instances
        self.test_ontology = test_ontology
        json.dump(self.test_ontology, open(test_ontology_file, "wt"), indent=4)
        json.dump([t.todict() for t in self.test], open(test_file, "wt"), indent=4)
        return

    def collect_instance_by_labels(self, labels:Iterable[Union[str, int, Tuple[str, str], Tuple[str, int]]], dataset:Union[str, None]=None) -> Dict[str, List[str]]:
        query = {}
        for label in labels:
            if dataset is None:
                dataset, label = label
            if dataset in query:
                query[dataset].add(label)
            else:
                query[dataset] = {label}
        response = {split: [] for split in self.splits}
        for dataset in query:
            data = getattr(self, dataset, None)
            if data is not None:
                for split in data:
                    response[split].extend([t for t in data[split] if t['label'] in query[dataset]])
        return response

    def feature_path(self, feature_path):
        return os.path.join(self.feature_root, feature_path)



class Batch(object):

    def __init__(self,
            token_ids: List[torch.LongTensor],
            spans: List[torch.LongTensor],
            labels:List[torch.LongTensor],
            # features: List[torch.FloatTensor],
            attention_masks:Union[List[torch.FloatTensor], None]=None,
            **kwargs)-> None:
        bsz = len(token_ids)
        assert len(labels) == bsz
        assert len(spans) == bsz
        assert all(len(x) == 2 for x in spans)
        if attention_masks is not None:
            assert len(attention_masks) == bsz
            assert all(len(x) == len(y) for x,y in zip(token_ids, attention_masks))
        _max_length = max(len(x) for x in token_ids)
        self.token_ids = torch.zeros(bsz, _max_length, dtype=torch.long)
        self.attention_masks = torch.zeros(bsz, _max_length, dtype=torch.float)
        for i in range(bsz):
            self.token_ids[i, :token_ids[i].size(0)] = token_ids[i]
            if attention_masks is not None:
                self.attention_masks[i, :token_ids[i].size(0)] = attention_masks[i]
            else:
                self.attention_masks[i, :token_ids[i].size(0)] = 1
        self.spans = torch.stack(spans, dim=0)
        self.label_mat = torch.stack(labels, dim=0)
        # self.features = torch.stack(features, dim=0)
        self.meta = kwargs

    def pin_memory(self):
        self.token_ids = self.token_ids.pin_memory()
        self.attention_masks = self.attention_masks.pin_memory()
        self.spans = self.spans.pin_memory()
        self.label_mat = self.label_mat.pin_memory()
        # self.features = self.features.pin_memory()
        return self

    def cuda(self,device:Union[torch.device,int,None]=None):
        assert torch.cuda.is_available()
        self.token_ids = self.token_ids.cuda(device)
        self.attention_masks = self.attention_masks.cuda(device)
        self.spans = self.spans.cuda(device)
        self.label_mat = self.label_mat.cuda(device)
        # self.features = self.features.cuda(device)
        return self

    def to(self, device:torch.device):
        self.token_ids = self.token_ids.to(device)
        self.attention_masks = self.attention_masks.to(device)
        self.spans = self.spans.to(device)
        self.label_mat = self.label_mat.to(device)
        # self.features = self.features.to(device)
        return self

    @classmethod
    def from_instances(cls, batch:List[Instance]):
        def slice(attr):
            return [getattr(t, attr) for t in batch]
        batch = [t.totensor() for t in batch]
        return cls(
            token_ids=slice("token_ids"),
            labels=slice("label"),
            # features=slice('features'),
            spans=slice("span"),
            sentence_ids=slice("sentence_id"),
            mention_ids=slice("mention_id"))

class LabelDataset(Dataset):
    def __init__(self, instances:List[Instance]) -> None:
        super().__init__()
        instances.sort(key=lambda i:i.label.index(1))
        self.label2index = {}
        i = 0
        labels = []
        for instance in instances:
            if len(labels) == 0 or instance.label.index(1) != labels[-1]:
                if len(labels) > 0:
                    self.label2index[labels[-1]] = (i, len(labels))
                i = len(labels)
            labels.append(instance.label.index(1))
        self.label2index[labels[-1]] = (i, len(labels))
        self.instances = instances

    def __len__(self) -> int:
        return len(self.instances)

    def __getitem__(self, index: int) -> Instance:
        instance = self.instances[index]
        return instance
        # return instance.load_clone()

    def get_indices_by_label(self, label:Tuple[str, str]) -> List[Instance]:
        return self.label2index[label]

    def collate_fn(self, batch:List[Instance]) -> Batch:
        return Batch.from_instances(batch)

def get_stage_loaders(root:str,
    batch_size:int,
    num_workers:int=0,
    seed:int=2147483647,
    *args,
    **kwargs):
    def prepare_dataset(instances:List[Dict]) -> List[Instance]:
        instances = [Instance(
            token_ids=instance["piece_ids"],
            span=instance["span"],
            features=collection.feature_path(instance["feature_path"]),
            sentence_id=instance["sentence_id"],
            mention_id=instance["mention_id"],
            label=collection.label2id[instance["label_onehot"]]
        ) for instance in instances]
        return instances

    collection = ZeroShotDataCollection(root)
    loaders = []
    dataset_train = LabelDataset(collection.train)
    train_loader = DataLoader(
        dataset=dataset_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=dataset_train.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed))
    loaders.append(train_loader)
    dataset_dev = LabelDataset(collection.dev)
    dev_loader = DataLoader(
        dataset=dataset_dev,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=dataset_dev.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed))
    loaders.append(dev_loader)
    dataset_test = LabelDataset(collection.test)
    test_loader = DataLoader(
        dataset=dataset_test,
        batch_size=batch_size,
        drop_last=False,
        collate_fn=dataset_test.collate_fn,
        pin_memory=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed))
    loaders.append(test_loader)

    onto = json.load(open(os.path.join(root, "ace", "ace.ontology_description.json")))
    labels = torch.LongTensor(onto["tokens"])
    label_masks = torch.BoolTensor(onto["masks"])

    return loaders, labels, label_masks


def test():
    l = get_stage_loaders(root="./data/", feature_root="/scratch/pengfei4/LInEx/data", batch_size=2, num_steps=5, episode_num_classes=4, episode_num_instances=3, episode_num_novel_classes=2, evaluation_num_instances=6)

if __name__ == "__main__":
    test()
