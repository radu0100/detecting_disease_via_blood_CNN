"""Build the tables and figures for the report and slides from runs/ and results/results.csv.

    python make_report.py              # baseline (resnet18) by default
    python make_report.py --model resnet18 --variant baseline

Outputs in report/:
    summary.csv, summary.md            mean +- std over seeds per task/model/variant
    data_<task>.png                    class counts per split
    curves_<task>.png                  train/val loss and val macro-F1 per epoch, all seeds
    cm_<task>.png                      row-normalised test confusion matrix (seed 0)
    qual_<task>.png                    correct and wrong test predictions (seed 0)
    per_class_<task>.csv               per-class precision / recall / F1 (seed 0)
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "report"
INK, MUTED, ACCENT, GRID = "#1f2a37", "#5b6b80", "#8a2432", "#ddd6c8"
SEED_COLORS = ["#33475b", "#7d93ab", "#b8a07e"]
plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
METRICS = ["macro_f1", "accuracy", "auroc", "sensitivity", "specificity", "patient_acc", "epochs", "train_time_min"]
TITLES = {"pbc": "PBC · cell type (8 classes)", "cnmc": "C-NMC 2019 · ALL vs healthy"}


def summary(res):
    res = res.drop_duplicates(["task", "model", "variant", "seed"], keep="last")
    g = res.groupby(["task", "model", "variant"])
    agg = g[METRICS].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    agg["n_seeds"] = g.size()
    agg = agg.reset_index()
    agg.to_csv(OUT / "summary.csv", index=False)
    lines = ["| task | model | variant | seeds | macro-F1 | accuracy | AUROC | sensitivity | specificity | patient acc |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in agg.iterrows():
        f = lambda m: "–" if pd.isna(r[f"{m}_mean"]) else f"{r[f'{m}_mean']:.3f} ± {0 if pd.isna(r[f'{m}_std']) else r[f'{m}_std']:.3f}"
        lines.append(f"| {r.task} | {r.model} | {r.variant} | {r.n_seeds} | {f('macro_f1')} | {f('accuracy')} | "
                     f"{f('auroc')} | {f('sensitivity')} | {f('specificity')} | {f('patient_acc')} |")
    (OUT / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def data_plot(task):
    df = pd.read_csv(ROOT / "data" / "splits" / f"{task}.csv")
    col = "split" if task == "pbc" else "split_patient"
    ct = pd.crosstab(df.label, df[col])[["train", "val", "test"]]
    ct = ct.loc[ct.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(ct) + 1.2))
    left = np.zeros(len(ct))
    for s, c in zip(["train", "val", "test"], SEED_COLORS):
        ax.barh(ct.index, ct[s], left=left, color=c, label=s, height=0.65)
        left += ct[s].values
    for i, v in enumerate(left):
        ax.text(v + left.max() * 0.01, i, f"{int(v):,}", va="center", fontsize=9, color=INK)
    ax.set_xlabel("images"); ax.set_title(TITLES[task], loc="left", color=INK)
    ax.legend(frameon=False, ncol=3, loc="lower right"); ax.grid(axis="x", color=GRID); ax.set_axisbelow(True)
    if "patient" in df:
        ax.text(0, -0.9, "patients: " + ", ".join(f"{k} {v}" for k, v in df.groupby(col).patient.nunique()
                [["train", "val", "test"]].items()), transform=ax.get_yaxis_transform(), fontsize=9, color=MUTED)
    fig.savefig(OUT / f"data_{task}.png"); plt.close(fig)


def curves(task, runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for i, r in enumerate(runs):
        h = pd.read_csv(r / "history.csv"); c = SEED_COLORS[i % 3]; s = json.loads((r / "metrics.json").read_text())["args"]["seed"]
        axes[0].plot(h.epoch, h.train_loss, color=c, lw=1.6, label=f"train · seed {s}")
        axes[0].plot(h.epoch, h.val_loss, color=c, lw=1.6, ls="--", label=f"val · seed {s}")
        axes[1].plot(h.epoch, h.val_macro_f1, color=c, lw=1.8, label=f"seed {s}")
        b = h.val_macro_f1.idxmax(); axes[1].scatter(h.epoch[b], h.val_macro_f1[b], color=ACCENT, zorder=3, s=22)
    axes[0].set(xlabel="epoch", ylabel="loss", title="Loss (train weighted, val unweighted)")
    axes[1].set(xlabel="epoch", ylabel="val macro-F1", title="Validation macro-F1 (dot = best checkpoint)")
    for ax in axes:
        ax.grid(color=GRID); ax.legend(frameon=False, fontsize=8); ax.title.set_color(INK)
    fig.suptitle(TITLES[task], x=0.01, y=1.03, ha="left", color=INK, fontsize=13)
    fig.savefig(OUT / f"curves_{task}.png"); plt.close(fig)


def confusion(task, run, classes):
    p = pd.read_csv(run / "test_preds.csv")
    cm = confusion_matrix(p.true, p.pred, labels=range(len(classes)))
    cmn = cm / cm.sum(1, keepdims=True)
    n = len(classes)
    fig, ax = plt.subplots(figsize=(1.0 * n + 2.2, 1.0 * n + 1.6) if n > 2 else (4.4, 3.8))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{cmn[i, j]:.2f}\n({cm[i, j]})", ha="center", va="center", fontsize=8 if n > 2 else 10,
                    color="white" if cmn[i, j] > 0.6 else INK)
    ax.set_xticks(range(n), classes, rotation=45 if n > 2 else 0, ha="right" if n > 2 else "center")
    ax.set_yticks(range(n), classes); ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"{TITLES[task]} · test, seed 0", loc="left", color=INK, fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)
    fig.savefig(OUT / f"cm_{task}.png"); plt.close(fig)
    rep = classification_report(p.true, p.pred, labels=range(n), target_names=classes, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.round(4).to_csv(OUT / f"per_class_{task}.csv")
    if task == "pbc" and "ig" in classes and "neutrophil" in classes:
        ig, ne = classes.index("ig"), classes.index("neutrophil")
        print(f"[pbc] IG->neutrophil {cmn[ig, ne]:.3f}   neutrophil->IG {cmn[ne, ig]:.3f}")


def qualitative(task, run, classes, size):
    p = pd.read_csv(run / "test_preds.csv")
    X = np.load(ROOT / "data" / "cache" / f"{task}_{size}.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    p["conf"] = p[[f"p_{c}" for c in classes]].max(axis=1)
    right = p[p.true == p.pred]
    wrong = p[p.true != p.pred].sort_values("conf", ascending=False)
    k = 6
    if task == "pbc":  # one correct example per class where possible
        right = right.groupby("true").sample(1, random_state=0).sample(frac=1, random_state=0)
    else:
        right = pd.concat([right[right.true == c].sample(min(k // 2, (right.true == c).sum()), random_state=0) for c in range(2)])
    right, wrong = right.head(k), wrong.head(k)
    fig, axes = plt.subplots(2, k, figsize=(2.0 * k, 4.9))
    for row, (sub, lab) in enumerate([(right, "correct"), (wrong, "wrong (most confident)")]):
        for j in range(k):
            ax = axes[row, j]; ax.axis("off")
            if j >= len(sub): continue
            r = sub.iloc[j]
            ax.imshow(X[int(r.row)])
            ax.set_title(f"true: {classes[int(r.true)]}\npred: {classes[int(r.pred)]} ({r.conf:.2f})", fontsize=8,
                         color=INK if row == 0 else ACCENT)
        axes[row, 0].text(-0.08, 0.5, lab, transform=axes[row, 0].transAxes, rotation=90, va="center", ha="right",
                          fontsize=10, color=INK)
    fig.suptitle(f"{TITLES[task]} · test examples, seed 0", x=0.01, ha="left", color=INK)
    fig.savefig(OUT / f"qual_{task}.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18")
    ap.add_argument("--variant", default="baseline")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    res_path = ROOT / "results" / "results.csv"
    if res_path.exists():
        summary(pd.read_csv(res_path))
    for task in ("pbc", "cnmc"):
        if (ROOT / "data" / "splits" / f"{task}.csv").exists():
            data_plot(task)
        runs = sorted((ROOT / "runs").glob(f"{task}_{a.model}_{a.variant}_s*"))
        runs = [r for r in runs if (r / "metrics.json").exists()]
        if not runs:
            continue
        meta = json.loads((runs[0] / "metrics.json").read_text())
        classes, size = meta["classes"], meta["args"]["size"]
        curves(task, runs)
        confusion(task, runs[0], classes)
        qualitative(task, runs[0], classes, size)
        if task == "cnmc":
            per_patient(runs)
    print(f"figures written to {OUT}")


def per_patient(runs):
    """C-NMC: cell accuracy per test patient, mean and range over seeds."""
    frames = []
    for r in runs:
        p = pd.read_csv(r / "test_preds.csv")
        frames.append(p.assign(ok=p.true == p.pred).groupby("patient").agg(
            true=("true", "first"), n=("true", "size"), acc=("ok", "mean"), p_all=("p_all", "mean")))
    d = pd.concat(frames).reset_index()
    s = d.groupby("patient").agg(true=("true", "first"), n=("n", "first"), acc=("acc", "mean"),
                                 lo=("acc", "min"), hi=("acc", "max"), p_all=("p_all", "mean"))
    s = s.sort_values(["true", "acc"])
    s.round(4).to_csv(OUT / "per_patient_cnmc.csv")
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(s))
    ax.bar(x, s.acc, color=[ACCENT if t == 0 else SEED_COLORS[0] for t in s.true], width=0.7)
    ax.errorbar(x, s.acc, yerr=[s.acc - s.lo, s.hi - s.acc], fmt="none", ecolor=MUTED, capsize=3, lw=1)
    for i, (n, a) in enumerate(zip(s.n, s.acc)):
        ax.text(i, a + 0.03, f"n={n}", ha="center", fontsize=8, color=MUTED)
    ax.set_xticks(x, [f"{p}\n{'healthy' if t == 0 else 'ALL'}" for p, t in zip(s.index, s.true)], fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("cell accuracy"); ax.grid(axis="y", color=GRID); ax.set_axisbelow(True)
    ax.set_title("C-NMC test patients · cell accuracy per patient (bar = mean of 3 seeds, whisker = range)",
                 loc="left", fontsize=10, color=INK)
    fig.savefig(OUT / "per_patient_cnmc.png"); plt.close(fig)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
