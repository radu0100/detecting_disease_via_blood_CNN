# Detecting Disease via Blood — blood-cell CNN benchmark

Fine-tunes an ImageNet-pretrained **ResNet-18** to classify single blood-cell images on two tasks:

- **PBC** — 8 normal cell types (17,092 images)
- **C-NMC 2019** — leukemic (ALL) vs healthy cells, tested on **patients the model has never seen** (10,311 cells, 72 patients)

| Task (test set, 3 seeds) | Macro-F1 | AUROC | Sensitivity | Specificity |
|---|---|---|---|---|
| PBC | 0.990 ± 0.000 | – | – | – |
| C-NMC | 0.821 ± 0.013 | 0.897 ± 0.010 | 0.968 ± 0.013 | 0.644 ± 0.042 |

## Quick start

You need Python 3.11+ and an NVIDIA GPU. Run these from the project folder (in VS Code: **Terminal › New Terminal**).

**1. Create an environment and install everything** (one time, ~3 GB download)
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
On macOS/Linux, activate with `source .venv/bin/activate` instead.
If PowerShell says running scripts is disabled, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or use `cmd` instead.

**2. Download and prepare the data** (one time, ~1.4 GB download, a few minutes)
```bash
python prepare_data.py
```

**3. Train**
```bash
python run_baseline.py
```
This trains ResNet-18 on both tasks with 3 seeds (~30 min on an RTX 5070 Ti laptop). Finished runs are skipped if you run it again.

**4. Make the tables and figures**
```bash
python make_report.py
```

Every time you open a new terminal, activate the environment again first (`.venv\Scripts\activate`).

## One run at a time

```bash
python train.py --task cnmc --seed 0
```
`--task pbc` or `--task cnmc`. Each run takes 3–6 minutes and prints one line per epoch, then the test scores.

Other experiments change one option:

| Experiment | Command |
|---|---|
| Deeper model | `python train.py --task cnmc --model resnet50` |
| Transformer | `python train.py --task cnmc --model deit_tiny_patch16_224 --lr 1e-4` |
| No augmentation | `python train.py --task cnmc --aug none` |
| Oversample instead of weighted loss | `python train.py --task cnmc --balance oversample` |
| Leakage check (random cell split) | `python train.py --task cnmc --split random` |
| Shape-only control | `python train.py --task cnmc --input mask` |
| Bigger images | `python prepare_data.py --sizes 384`, then add `--size 384` |

Any [timm](https://huggingface.co/timm) model name works with `--model`.

## Where results go

| Path | Contents |
|---|---|
| `results/results.csv` | One row of test scores per run (added automatically) |
| `runs/<run name>/` | Saved model (`best.pt`), per-epoch history, test predictions. The 6 baseline models are included in the repo |
| `report/` | Figures and `summary.md` after `make_report.py` |

## Data notes

- `prepare_data.py` downloads public Hugging Face copies: [chehablab/PBC](https://huggingface.co/datasets/chehablab/PBC) (Acevedo et al., 2020) and [dwb2023/cnmc-leukemia-2019](https://huggingface.co/datasets/dwb2023/cnmc-leukemia-2019) (C-NMC 2019 training folds). To use your own copy, put it in `data/raw/PBC/` or `data/raw/C-NMC/` before running it.
- 350 C-NMC cells are stored as all-zero bytes in that published copy and are skipped automatically (patient 67: all 267 cells; patient 49: 66; patient 68: 17). They are listed in `data/splits/excluded_C-NMC.csv`.
- The C-NMC preliminary test set (1,867 cells) is not in that copy.

## Protocol

| | |
|---|---|
| Splits | 70/15/15, seed 42. PBC by image, stratified by class. C-NMC by **patient** (50 / 11 / 11 patients) |
| Input | 224×224 RGB, ImageNet normalization; flips, 90° rotations, light color jitter |
| Training | AdamW, lr 3e-4, weight decay 0.05, cosine schedule, batch 64, bf16, class-weighted loss |
| Stopping | Up to 50 epochs; best validation macro-F1, patience 10; test evaluated once |
| Seeds | 0, 1, 2 |
| Metrics | Macro-F1; for C-NMC also AUROC, sensitivity, specificity (ALL = positive), patient-level accuracy |
