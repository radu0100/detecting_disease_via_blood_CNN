# Detecting Disease via Blood — Phase 1 baseline

Patient-level benchmark for blood-cell image classification: PBC (8 normal cell types) and
C-NMC 2019 (acute lymphoblastic leukemia vs healthy). Baseline: ImageNet-pretrained ResNet-18.

## Setup
```bash
py -3.14 -m venv .venv
.venv/Scripts/python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install scikit-learn pandas matplotlib pillow tqdm timm pyarrow
```
RTX 50xx (Blackwell) GPUs need a CUDA 12.8+ PyTorch build.

## Data
Put the Hugging Face parquet exports (or the original image folders) under `data/raw/`:
- `data/raw/PBC/` — PBC, Acevedo et al. 2020 (Mendeley `snkd93bnjr`). 17,092 images, 8 classes.
- `data/raw/C-NMC/` — C-NMC 2019 training folds (HF `dwb2023/cnmc-leukemia-2019`). 10,661 cells, 73 patients.

**Data notes**
- The HF C-NMC export contains only the 3 training folds; the 1,867-cell preliminary test set is not included.
- 350 C-NMC cells in shard `train-00010` are stored as all-zero bytes in the published export
  (the shard matches its published SHA-256, so this is not a download error). They are excluded:
  patient 67 (all 267 cells), patient 49 (66), patient 68 (17). List: `data/splits/excluded_C-NMC.csv`.
  **Usable: 10,311 cells from 72 patients.**

## Protocol (everyone uses the same)
| | |
|---|---|
| Splits | 70/15/15, `random_state=42`. PBC: stratified by class (no patient IDs). C-NMC: by **patient**, stratified by patient class → 50/11/11 patients |
| Input | 224×224 RGB, ImageNet normalization |
| Augmentation | flips, 90° rotations, light color jitter (on GPU) |
| Optimizer | AdamW, lr 3e-4, wd 0.05, cosine schedule, batch 64, bf16 autocast |
| Loss | cross-entropy weighted by inverse class frequency |
| Stopping | max 50 epochs, early stopping on val macro-F1 (patience 10), best checkpoint evaluated once on test |
| Seeds | 0, 1, 2 |
| Metrics | macro-F1, accuracy; C-NMC also AUROC, sensitivity, specificity (ALL = positive), patient-level accuracy (mean cell probability > 0.5) |

## Run
```bash
.venv/Scripts/python prepare_data.py          # splits + 224 px cache (add --sizes 128 384 for ablations)
.venv/Scripts/python run_baseline.py          # ResNet-18, 2 tasks × 3 seeds (~30 min on an RTX 5070 Ti laptop)
.venv/Scripts/python make_report.py           # tables + figures in report/
```
Other experiments use `train.py` switches, e.g.
`--model resnet50`, `--model deit_tiny_patch16_224 --lr 1e-4`, `--aug none|strong`,
`--balance plain|oversample`, `--size 384`, `--split random` (leakage control), `--input mask` (shape-only control).
Every run appends one row to `results/results.csv`.
