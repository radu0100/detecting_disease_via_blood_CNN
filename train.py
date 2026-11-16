"""Train one configuration on one task with one seed.

Examples:
    python train.py --task pbc  --model resnet18 --seed 0
    python train.py --task cnmc --model resnet18 --seed 0
    python train.py --task cnmc --model resnet18 --seed 0 --split random   # leakage control
    python train.py --task pbc  --model deit_tiny_patch16_224 --lr 1e-4    # any timm model

Images are pre-resized (prepare_data.py) and augmented on the GPU, so no DataLoader workers are needed.
Writes runs/<run_name>/{history.csv, metrics.json, test_preds.csv, best.pt} and appends to results/results.csv.
"""
import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULT_COLS = ["person", "task", "model", "variant", "seed", "macro_f1", "accuracy", "auroc",
               "sensitivity", "specificity", "epochs", "best_epoch", "train_time_min", "patient_acc"]


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["pbc", "cnmc"], required=True)
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--aug", choices=["default", "none", "strong"], default="default")
    ap.add_argument("--balance", choices=["weighted", "plain", "oversample"], default="weighted")
    ap.add_argument("--split", choices=["patient", "random"], default="patient", help="C-NMC only")
    ap.add_argument("--input", choices=["rgb", "mask"], default="rgb", help="mask = silhouette-only control")
    ap.add_argument("--person", default="me")
    ap.add_argument("--tag", default="", help="extra text for the variant name")
    return ap.parse_args(argv)


def variant_name(a):
    parts = []
    if a.size != 224: parts.append(f"size{a.size}")
    if a.aug != "default": parts.append(f"aug-{a.aug}")
    if a.balance != "weighted": parts.append(f"bal-{a.balance}")
    if a.task == "cnmc" and a.split != "patient": parts.append(f"split-{a.split}")
    if a.input != "rgb": parts.append(f"input-{a.input}")
    if a.lr != 3e-4: parts.append(f"lr{a.lr:g}")
    if a.tag: parts.append(a.tag)
    return "_".join(parts) or "baseline"


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ---------------------------------------------------------------- data
def load_task(a):
    df = pd.read_csv(DATA / "splits" / f"{a.task}.csv")
    classes = json.loads((DATA / "splits" / f"{a.task}_classes.json").read_text())
    X = torch.from_numpy(np.load(DATA / "cache" / f"{a.task}_{a.size}.npy"))  # N,H,W,3 uint8 in RAM
    y = torch.tensor(df.label_idx.values, dtype=torch.long)
    col = "split" if a.task == "pbc" else f"split_{a.split}"
    idx = {s: np.where(df[col].values == s)[0] for s in ("train", "val", "test")}
    return df, classes, X, y, idx


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def to_input(xb, a, train):
    """xb: B,H,W,3 uint8 on GPU -> normalized B,3,H,W float."""
    x = xb.permute(0, 3, 1, 2).float().div_(255)
    if a.input == "mask":  # cell silhouette: anything noticeably brighter than the black background
        x = (x.amax(1, keepdim=True) > 0.08).float().expand(-1, 3, -1, -1).contiguous()
    B, dev = x.shape[0], x.device
    if train and a.aug != "none":
        m = torch.rand(B, device=dev) < 0.5; x[m] = x[m].flip(3)
        m = torch.rand(B, device=dev) < 0.5; x[m] = x[m].flip(2)
        k = torch.randint(0, 4, (B,), device=dev)
        for r in (1, 2, 3):
            m = k == r
            if m.any(): x[m] = torch.rot90(x[m], r, (2, 3))
        if a.input == "rgb":  # light color jitter
            u = lambda: 1 + (torch.rand(B, 1, 1, 1, device=dev) * 2 - 1) * 0.1
            x = x * u()
            mu = x.mean((1, 2, 3), keepdim=True); x = (x - mu) * u() + mu
            g = (0.299 * x[:, :1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:]); x = (x - g) * u() + g
            x = x.clamp_(0, 1)
        if a.aug == "strong":  # random resized crop (zoom 1-1.33x, shift) + occasional blur
            s = torch.empty(B, device=dev).uniform_(0.75, 1.0)
            t = (torch.rand(B, 2, device=dev) * 2 - 1) * (1 - s)[:, None]
            theta = torch.zeros(B, 2, 3, device=dev)
            theta[:, 0, 0], theta[:, 1, 1], theta[:, :, 2] = s, s, t
            x = F.grid_sample(x, F.affine_grid(theta, x.shape, align_corners=False), align_corners=False)
            m = torch.rand(B, device=dev) < 0.3
            if m.any():
                from torchvision.transforms.v2.functional import gaussian_blur
                x[m] = gaussian_blur(x[m], kernel_size=[5, 5], sigma=[1.0, 1.0])
    return (x - MEAN.to(dev)) / STD.to(dev)


# ---------------------------------------------------------------- model
def build_model(name, n_classes):
    if name in ("resnet18", "resnet34", "resnet50"):
        import torchvision.models as tvm
        w = {"resnet18": tvm.ResNet18_Weights.IMAGENET1K_V1,
             "resnet34": tvm.ResNet34_Weights.IMAGENET1K_V1,
             "resnet50": tvm.ResNet50_Weights.IMAGENET1K_V2}[name]
        m = getattr(tvm, name)(weights=w)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
        return m
    import timm
    return timm.create_model(name, pretrained=True, num_classes=n_classes)


@torch.no_grad()
def predict(model, X, y, idx, a, dev, crit, bs=256):
    model.eval()
    probs, loss_sum = [], 0.0
    for i in range(0, len(idx), bs):
        b = torch.as_tensor(idx[i:i + bs])
        xb = to_input(X[b].to(dev, non_blocking=True), a, train=False)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(xb).float()
        loss_sum += crit(out, y[b].to(dev)).item() * len(b)
        probs.append(out.softmax(1).cpu())
    return torch.cat(probs).numpy(), loss_sum / len(idx)


def metrics(y_true, prob, task):
    pred = prob.argmax(1)
    m = {"macro_f1": f1_score(y_true, pred, average="macro"), "accuracy": accuracy_score(y_true, pred)}
    if task == "cnmc":
        tp = ((pred == 1) & (y_true == 1)).sum(); fn = ((pred == 0) & (y_true == 1)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum(); fp = ((pred == 1) & (y_true == 0)).sum()
        m.update(auroc=roc_auc_score(y_true, prob[:, 1]),
                 sensitivity=tp / max(tp + fn, 1), specificity=tn / max(tn + fp, 1))
    return {k: float(v) for k, v in m.items()}


# ---------------------------------------------------------------- main
def main(argv=None):
    a = get_args(argv)
    set_seed(a.seed)
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    variant = variant_name(a)
    run = f"{a.task}_{a.model}_{variant}_s{a.seed}"
    out_dir = ROOT / "runs" / run
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== {run}")

    df, classes, X, y, idx = load_task(a)
    tr, va, te = idx["train"], idx["val"], idx["test"]
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  classes {classes}")

    counts = np.bincount(y[tr].numpy(), minlength=len(classes))
    w = torch.tensor(len(tr) / (len(classes) * counts), dtype=torch.float32, device=dev)
    crit_train = nn.CrossEntropyLoss(weight=w if a.balance == "weighted" else None)
    crit_eval = nn.CrossEntropyLoss()
    sample_w = (1.0 / counts)[y[tr].numpy()] if a.balance == "oversample" else None

    model = build_model(a.model, len(classes)).to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(tr) // a.bs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * steps)
    gen = torch.Generator().manual_seed(a.seed)

    hist, best_f1, best_ep, best_state, bad = [], -1.0, 0, None, 0
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        te0 = time.time()
        if sample_w is not None:
            order = tr[torch.multinomial(torch.tensor(sample_w), len(tr), replacement=True, generator=gen).numpy()]
        else:
            order = tr[torch.randperm(len(tr), generator=gen).numpy()]
        loss_sum, correct, seen = 0.0, 0, 0
        for i in range(steps):
            b = torch.as_tensor(order[i * a.bs:(i + 1) * a.bs])
            xb = to_input(X[b].to(dev, non_blocking=True), a, train=True).contiguous(memory_format=torch.channels_last)
            yb = y[b].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
                loss = crit_train(out.float(), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step(); sched.step()
            loss_sum += loss.item() * len(b); correct += (out.argmax(1) == yb).sum().item(); seen += len(b)

        pv, vloss = predict(model, X, y, va, a, dev, crit_eval)
        vm = metrics(y[va].numpy(), pv, a.task)
        row = {"epoch": ep, "train_loss": loss_sum / seen, "train_acc": correct / seen, "val_loss": vloss,
               "val_acc": vm["accuracy"], "val_macro_f1": vm["macro_f1"], "lr": opt.param_groups[0]["lr"],
               "epoch_sec": time.time() - te0}
        if a.task == "cnmc": row["val_auroc"] = vm["auroc"]
        hist.append(row)
        print(f"ep {ep:2d}  loss {row['train_loss']:.4f}  val_loss {vloss:.4f}  val_f1 {vm['macro_f1']:.4f}"
              f"  ({row['epoch_sec']:.0f}s)", flush=True)

        if vm["macro_f1"] > best_f1:
            best_f1, best_ep, bad = vm["macro_f1"], ep, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= a.patience:
                print(f"early stop at epoch {ep} (best {best_ep})")
                break
    train_min = (time.time() - t0) / 60
    pd.DataFrame(hist).to_csv(out_dir / "history.csv", index=False)

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    pt, _ = predict(model, X, y, te, a, dev, crit_eval)
    yt = y[te].numpy()
    m = metrics(yt, pt, a.task)

    preds = pd.DataFrame({"row": te, "true": yt, "pred": pt.argmax(1)})
    for c, name in enumerate(classes):
        preds[f"p_{name}"] = pt[:, c]
    if a.task == "cnmc":
        preds["patient"] = df.patient.values[te]
        pat = preds.groupby("patient").agg(true=("true", "first"), p=("p_all", "mean"))
        m["patient_acc"] = float(((pat.p > 0.5).astype(int) == pat.true).mean())
    preds.to_csv(out_dir / "test_preds.csv", index=False)

    m.update(epochs=len(hist), best_epoch=best_ep, train_time_min=round(train_min, 2))
    meta = {"run": run, "variant": variant, "args": vars(a), "classes": classes, "test": m,
            "n": {"train": len(tr), "val": len(va), "test": len(te)}}
    (out_dir / "metrics.json").write_text(json.dumps(meta, indent=2))

    res = ROOT / "results" / "results.csv"
    res.parent.mkdir(exist_ok=True)
    row = {"person": a.person, "task": a.task, "model": a.model, "variant": variant, "seed": a.seed, **m}
    pd.DataFrame([row]).reindex(columns=RESULT_COLS).to_csv(res, mode="a", header=not res.exists(), index=False)
    print("TEST", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}))
    return m


if __name__ == "__main__":
    main()
