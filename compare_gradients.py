import json
from models.nets_span import IEToken, IESPAN, IEFromNLI
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.data import get_data, get_example_dataset
from utils.options import parse_arguments
from utils.worker import Worker

def zero_grad(model):
    for param in model.parameters():
        if param.grad is not None:
            param.grad = None

def compute_inner_product(model, ref_grads):
    dot = 0
    for name, param in model.named_parameters():
        if param.requires_grad and name in ref_grads and ref_grads[name] is not None:
                dot += torch.sum(param.grad.data * ref_grads[name])
    return dot


def main():

    opts = parse_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opts.gpu}"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    if opts.seed == -1:
        import time
        opts.seed = time.time()
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    if opts.gpu.count(",") > 0:
        opts.batch_size = opts.batch_size * (opts.gpu.count(",")+1)
        opts.eval_batch_size = opts.eval_batch_size * (opts.gpu.count(",")+1)
    loaders, label2id = get_data(opts, batch_size=1, shuffle=False)
    example_dataset = get_example_dataset(opts)
    example_loader = DataLoader(
        dataset=example_dataset,
        batch_size=opts.batch_size,
        collate_fn=example_dataset.collate_fn,
        pin_memory=True
    )

    if opts.setting == 'span':
        IEModel = IESPAN
    elif opts.setting == "token":
        IEModel = IEToken
    elif opts.setting == "nli":
        IEModel = IEFromNLI
    elif opts.setting == "sentence":
        IEModel = IESentence

    model = IEModel(
        hidden_dim=opts.hidden_dim,
        nclass=len(label2id) if opts.setting != "sentence" else len(label2id) - 1,
        model_name=opts.model_name,
        distributed=opts.gpu.count(",") > 0
    )

    if opts.gpu.count(",") > 0:
        model = torch.nn.DataParallel(model)

    model.to(torch.device('cuda:0') if torch.cuda.is_available() and (not opts.no_gpu) else torch.device('cpu'))

    model.load_state_dict(torch.load(opts.load_model)['state_dict'])
    model.eval()
    
    for example_batch in tqdm(example_loader):
        if not opts.no_gpu and not opts.gpu.count(",") > 0:
            example_batch = Worker._to_device(example_batch, torch.device(f"cuda:0"))
        output = model.forward(example_batch)
        if isinstance(output, dict):
            loss = output.pop("loss")
        else:
            loss = output
        if len(loss.size()) >= 1:
            loss = loss.sum()
        else:
            loss = loss * len(example_batch)
        loss = loss / len(example_dataset)
        loss.backward()
    grads = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads[name] = param.grad.data
    zero_grad(model)

    weakly_supervised_data_file = os.path.join(opts.root, opts.weak_corpus, f"weakly_supervised_data_{opts.weak_annotation}.jsonl")
    with open(weakly_supervised_data_file, "rt") as fp:
        ws_data = [json.loads(t) for t in fp]
    tokenizer = AutoTokenizer.from_pretrained(opts.model_name)
    discarded = []
    for idx, train_batch in enumerate(loaders[0]):
        if not opts.no_gpu and not opts.gpu.count(",") > 0:
            train_batch = Worker._to_device(train_batch, torch.device(f"cuda:0"))
        output = model.forward(train_batch)
        if isinstance(output, dict):
            loss = output.pop("loss")
        else:
            loss = output
        if len(loss.size()) >= 1:
            loss = loss.sum()
        else:
            loss = loss * len(example_batch)
        loss = loss / len(example_dataset)
        loss.backward()
        dot = compute_inner_product(model, grads)
        if dot < 0:
            discarded.append(ws_data[idx])
            print(ws_data[idx])
        zero_grad(model)
    json.dump(discarded, open("data/discarded_ws.json", "wt"), indent=4)
        



if __name__ == "__main__":
    main()
