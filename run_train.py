import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import json
import os
from tqdm import tqdm

from utils.optimizer import AdamW
from utils.options import parse_arguments
from utils.datastream import get_stage_loaders
from utils.worker import Worker
from models.nets import ZIE

PERM = [[0, 1, 2, 3,4], [4, 3, 2, 1, 0], [0, 3, 1, 4, 2], [1, 2, 0, 3, 4], [3, 4, 0, 1, 2]]

def add_summary_value(writer, key, value, iteration):
    if writer:
        writer.add_scalar(key, value, iteration)


def by_class(preds, labels, learned_labels=None):
    match = (preds == labels).float()
    nlabels = max(torch.max(labels).item(), torch.max(preds).item())
    bc = {}

    ag = 0; ad = 0; am = 0
    for label in range(1, nlabels+1):
        lg = (labels==label); ld = (preds==label)
        lr = torch.sum(match[lg]) / torch.sum(lg.float())
        lp = torch.sum(match[ld]) / torch.sum(ld.float())
        lf = 2 * lr * lp / (lr + lp)
        if torch.isnan(lf):
            bc[label] = (0, 0, 0)
        else:
            bc[label] = (lp.item(), lr.item(), lf.item())
        if learned_labels is not None and label in learned_labels:
            ag += lg.float().sum()
            ad += ld.float().sum()
            am += match[lg].sum()
    if learned_labels is None:
        ag = (labels!=0); ad = (preds!=0)
        sum_ad = torch.sum(ad.float())
        if sum_ad == 0:
            ap = ar = 0
        else:
            ar = torch.sum(match[ag]) / torch.sum(ag.float())
            ap = torch.sum(match[ad]) / torch.sum(ad.float())
    else:
        if ad == 0:
            ap = ar = 0
        else:
            ar = am / ag; ap = am / ad
    if ap == 0:
        af = ap = ar = 0
    else:
        af = 2 * ar * ap / (ar + ap)
        af = af.item(); ar = ar.item(); ap = ap.item()
    return bc, (ap, ar, af)


def main():

    opts = parse_arguments()
    torch.manual_seed(opts.seed)
    np.random.seed(opts.seed)
    summary = SummaryWriter(opts.log_dir)


    loaders, labels, label_masks = get_stage_loaders(root=opts.json_root,
        batch_size=opts.batch_size,
        num_workers=8)


    model = ZIE(
        input_dim=opts.input_dim,
        hidden_dim=opts.hidden_dim,
        device=torch.device(torch.device(f'cuda:{opts.gpu}' if torch.cuda.is_available() and (not opts.no_gpu) else 'cpu'))
    )
    model.set_labels(labels, label_masks)

    param_groups = [
        {"params": [param for name, param in model.named_parameters() if param.requires_grad and 'bert' not in name],
        "lr":opts.learning_rate,
        "weight_decay": opts.decay,
        "betas": (0.9, 0.999)},
        {"params": [param for name, param in model.named_parameters() if param.requires_grad and 'bert' in name],
        "lr":opts.bert_learning_rate,
        "weight_decay": opts.bert_decay,
        "betas": (0.9, 0.98)}
        ]
    optimizer = AdamW(params=param_groups)
    worker = Worker(opts)
    worker._log(str(opts))
    if opts.test_only:
        worker.load(model)
    best_dev = best_test = None
    collect_stats = "accuracy"
    collect_outputs = {"prediction", "label"}
    termination = False
    patience = opts.patience
    no_better = 0
    loader_id = 0
    total_epoch = 0

    dev_metrics = None
    test_metrics = None
    while not termination:
        if not opts.test_only:
            train_loss = lambda batch:model.forward(batch)
            epoch_loss, epoch_metric = worker.run_one_epoch(
                model=model,
                f_loss=train_loss,
                loader=loaders[loader_id],
                split="train",
                optimizer=optimizer,
                collect_stats=collect_stats,
                prog=loader_id)
            total_epoch += 1

            for output_log in [print, worker._log]:
                output_log(
                    f"Epoch {worker.epoch:3d}  Train Loss {epoch_loss} {epoch_metric}")
        else:

            termination = True

        score_fn = model.forward
        dev_loss, dev_metrics = worker.run_one_epoch(
            model=model,
            f_loss=score_fn,
            loader=loaders[-2],
            split="dev",
            collect_stats=collect_stats,
            collect_outputs=collect_outputs)
        dev_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
        dev_scores, (dev_p, dev_r, dev_f) = by_class(dev_outputs["prediction"], dev_outputs["label"])
        dev_class_f1 = {k: dev_scores[k][2] for k in dev_scores}
        for k,v in dev_class_f1.items():
            add_summary_value(summary, f"dev_class_{k}", v, total_epoch)
        dev_metrics = dev_f
        for output_log in [print, worker._log]:
            output_log(
                f"Epoch {worker.epoch:3d}:  Dev {dev_metrics}"
            )
        test_loss, test_metrics = worker.run_one_epoch(
            model=model,
            f_loss=score_fn,
            loader=loaders[-1],
            split="test",
            collect_stats=collect_stats,
            collect_outputs=collect_outputs)
        test_outputs = {k: torch.cat(v, dim=0) for k,v in worker.epoch_outputs.items()}
        torch.save(test_outputs, f"log/{os.path.basename(opts.load_model)}.output")
        test_scores, (test_p, test_r, test_f) = by_class(test_outputs["prediction"], test_outputs["label"])
        test_class_f1 = {k: test_scores[k][2] for k in test_scores}
        for k,v in test_class_f1.items():
            add_summary_value(summary, f"test_class_{k}", v, total_epoch)
        test_metrics = test_f
        for output_log in [print, worker._log]:
            output_log(
                f"Epoch {worker.epoch:3d}: Test {test_metrics}"
            )
        torch.save(test_outputs, "log/outputs")

        if not opts.test_only:
            if best_dev is None or dev_metrics > best_dev:
                best_dev = dev_metrics
                worker.save(model, optimizer, postfix=str(loader_id))
                best_test = test_metrics
                no_better = 0
            else:
                no_better += 1
            print(f"patience: {no_better} / {patience}")

            if (no_better == patience) or (worker.epoch == worker.train_epoch):
                loader_id += 1
                no_better = 0
                for output_log in [print, worker._log]:
                    output_log(f"BEST DEV {loader_id-1}: {best_dev if best_dev is not None else 0}")
                    output_log(f"BEST TEST {loader_id-1}: {best_test if best_test is not None else 0}")
                termination = True

if __name__ == "__main__":
    main()
