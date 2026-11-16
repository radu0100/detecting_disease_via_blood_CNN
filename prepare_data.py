"""Scan the raw datasets, build deterministic splits, and cache resized images.

Usage:
    python prepare_data.py                 # both tasks, 224 px
    python prepare_data.py --sizes 128 224 384

If data/raw/PBC or data/raw/C-NMC is empty, the public Hugging Face copy is downloaded first.

Expected raw layout (anything nested deeper is found by recursive search):
    data/raw/PBC/...<class>/*.jpg          class folders: basophil, eosinophil, erythroblast,
                                           ig, lymphocyte, monocyte, neutrophil, platelet
    data/raw/C-NMC/...                     training_data/fold_*/{all,hem}/*.bmp
                                           + preliminary test images and *labels*.csv

Outputs:
    data/splits/pbc.csv, data/splits/cnmc.csv, data/splits/<task>_classes.json
    data/cache/<task>_<size>.npy           uint8 array N x S x S x 3, rows in CSV order
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RAW = DATA / "raw"
SEED = 42

PBC_CLASSES = ["basophil", "eosinophil", "erythroblast", "ig",
               "lymphocyte", "monocyte", "neutrophil", "platelet"]
CNMC_CLASSES = ["hem", "all"]  # index 1 = leukemic (ALL) = positive class


def _valid(p: Path) -> bool:
    return not p.name.startswith("._") and "__MACOSX" not in p.parts


# Public Hugging Face copies used when data/raw/<name> is empty.
HF_SOURCES = {"PBC": "chehablab/PBC", "C-NMC": "dwb2023/cnmc-leukemia-2019"}


def ensure_raw(name: str):
    """Download a dataset from Hugging Face if its raw folder has no data yet (~0.3 GB PBC, ~1.1 GB C-NMC)."""
    root = RAW / name
    has_data = root.exists() and any(
        p.suffix.lower() in (".parquet", ".jpg", ".jpeg", ".png", ".bmp")
        for p in root.rglob("*") if ".git" not in p.parts)
    if has_data:
        return
    from huggingface_hub import snapshot_download
    print(f"  downloading {HF_SOURCES[name]} -> {root} ...", flush=True)
    snapshot_download(repo_id=HF_SOURCES[name], repo_type="dataset", local_dir=root,
                      allow_patterns=["data/*.parquet", "README.md"])


# ---------------------------------------------------------------- PBC
def _parquet_meta(root: Path) -> pd.DataFrame:
    """Non-image columns of every parquet shard under root, plus a 'path' = parquet:<file>#<row> pointer."""
    import pyarrow.parquet as pq
    files = [f for f in sorted(root.rglob("*.parquet")) if _valid(f) and ".git" not in f.parts]
    parts = []
    for f in files:
        pf = pq.ParquetFile(f)
        cols = [c for c in pf.schema_arrow.names if c != "image"]
        t = pq.read_table(f, columns=cols).to_pandas()
        t["path"] = [f"parquet:{f.relative_to(ROOT).as_posix()}#{i}" for i in range(len(t))]
        # Header check: some rows in the published C-NMC export are all-zero bytes (not images).
        ok = []
        for b in pf.iter_batches(columns=["image"], batch_size=512):
            for d in b.column("image").to_pylist():
                x = d["bytes"] or b""
                ok.append(x[:2] == b"BM" or x[:3] == b"\xff\xd8\xff" or x[:4] == b"\x89PNG")
        t["valid_image"] = ok
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    bad = df[~df.valid_image]
    if len(bad):
        DATA.joinpath("splits").mkdir(parents=True, exist_ok=True)
        bad.drop(columns="valid_image").to_csv(DATA / "splits" / f"excluded_{root.name}.csv", index=False)
        by = bad.subject_id.value_counts().to_dict() if "subject_id" in bad else {}
        print(f"  ! excluded {len(bad)} unreadable images from {root.name} {by}")
    return df[df.valid_image].drop(columns="valid_image").reset_index(drop=True)


def scan_pbc() -> pd.DataFrame:
    root = RAW / "PBC"
    pq_df = _parquet_meta(root)
    if not pq_df.empty:  # Hugging Face parquet export
        pq_df = pq_df.drop_duplicates("image_id").reset_index(drop=True)
        df = pd.DataFrame({"path": pq_df.path, "label": pq_df.cell_type.str.lower(),
                           "image_id": pq_df.image_id})
        df["label_idx"] = df.label.map(PBC_CLASSES.index)
        return _split_pbc(df)
    rows, seen = [], set()
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png") or not _valid(p):
            continue
        cls = p.parent.name.lower()
        if cls not in PBC_CLASSES or (cls, p.name) in seen:
            continue
        seen.add((cls, p.name))
        rows.append({"path": str(p.relative_to(ROOT)), "label": cls,
                     "label_idx": PBC_CLASSES.index(cls)})
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No PBC images found under {root}")
    return _split_pbc(df)


def _split_pbc(df: pd.DataFrame) -> pd.DataFrame:
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=df.label, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=df.label.iloc[rest], random_state=SEED)
    df["split"] = ""
    df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
    return df


# ---------------------------------------------------------------- C-NMC
_UID = re.compile(r"UID_(H?\d+)_", re.IGNORECASE)


def _patient(name: str):
    m = _UID.search(str(name))
    return m.group(1).upper() if m else None


def scan_cnmc() -> pd.DataFrame:
    root = RAW / "C-NMC"
    pq_df = _parquet_meta(root)
    if not pq_df.empty:  # Hugging Face parquet export (training folds only)
        pq_df = pq_df.drop_duplicates("original_image_name").reset_index(drop=True)
        cls = pq_df.class_label.str.lower()
        df = pd.DataFrame({"path": pq_df.path, "label": cls, "label_idx": cls.map(CNMC_CLASSES.index),
                           "patient": [str(s).upper() for s in pq_df.subject_id],
                           "orig_name": pq_df.original_image_name, "fold": pq_df.fold, "source": "train_folds"})
        print(f"  C-NMC (parquet): {len(df)} cells from {df.patient.nunique()} patients")
        return _split_cnmc(df)
    rows, seen = [], set()

    # Training folds: label from the all/hem folder, patient from the UID_ filename.
    for p in sorted(root.rglob("*.bmp")):
        if not _valid(p) or p.parent.name.lower() not in CNMC_CLASSES:
            continue
        pid = _patient(p.name)
        if pid is None or p.name in seen:
            continue
        seen.add(p.name)
        cls = p.parent.name.lower()
        rows.append({"path": str(p.relative_to(ROOT)), "label": cls,
                     "label_idx": CNMC_CLASSES.index(cls), "patient": pid,
                     "orig_name": p.name, "source": "train_folds"})
    n_train = len(rows)

    # Preliminary test set: anonymised file names + a labels CSV that keeps the original name.
    for csv_path in sorted(root.rglob("*.csv")):
        if not _valid(csv_path) or "label" not in csv_path.name.lower():
            continue
        lab = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in lab.columns}
        new_c = next((cols[c] for c in cols if "new" in c), None)
        orig_c = next((cols[c] for c in cols if "patient" in c or "orig" in c), None)
        lab_c = next((cols[c] for c in cols if c.startswith("label")), None)
        if not (new_c and orig_c and lab_c):
            print(f"  ! skipping {csv_path.name}: columns {list(lab.columns)}")
            continue
        # Final-phase test images are also named 1.bmp, 2.bmp, ... so only look in "prelim" folders.
        img_dirs = [d for d in root.rglob("*") if d.is_dir() and _valid(d) and "prelim" in d.name.lower()]
        if not img_dirs:
            img_dirs = [csv_path.parent] + [d for d in csv_path.parent.rglob("*") if d.is_dir() and _valid(d)]
        lookup = {}
        for d in img_dirs:
            for f in d.glob("*.bmp"):
                lookup.setdefault(f.name, f)
        for _, r in lab.iterrows():
            f = lookup.get(str(r[new_c]))
            pid = _patient(r[orig_c])
            if f is None or pid is None or str(r[orig_c]) in seen:
                continue
            seen.add(str(r[orig_c]))
            li = int(r[lab_c])  # 1 = ALL, 0 = hem
            rows.append({"path": str(f.relative_to(ROOT)), "label": CNMC_CLASSES[li],
                         "label_idx": li, "patient": pid,
                         "orig_name": str(r[orig_c]), "source": "prelim_test"})
    print(f"  C-NMC: {n_train} training-fold cells, {len(rows) - n_train} preliminary-test cells")

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No C-NMC images found under {root}")
    return _split_cnmc(df)


def _split_cnmc(df: pd.DataFrame) -> pd.DataFrame:

    # Patient-level split (main protocol), stratified by the patient's class.
    pat = df.groupby("patient").label_idx.agg(lambda s: int(s.mode().iloc[0])).sort_index()
    pids = np.asarray(pat.index, dtype=object)
    p_tr, p_rest = train_test_split(pids, test_size=0.30, stratify=pat.values, random_state=SEED)
    p_va, p_te = train_test_split(p_rest, test_size=0.50, stratify=pat.loc[p_rest].values, random_state=SEED)
    pmap = {**{p: "train" for p in p_tr}, **{p: "val" for p in p_va}, **{p: "test" for p in p_te}}
    df["split_patient"] = df.patient.map(pmap)

    # Cell-level random split (leakage control only).
    idx = np.arange(len(df))
    tr, rest = train_test_split(idx, test_size=0.30, stratify=df.label_idx, random_state=SEED)
    va, te = train_test_split(rest, test_size=0.50, stratify=df.label_idx.iloc[rest], random_state=SEED)
    df["split_random"] = ""
    df.loc[tr, "split_random"], df.loc[va, "split_random"], df.loc[te, "split_random"] = "train", "val", "test"
    return df


# ---------------------------------------------------------------- caching
def _load(path: str, size: int) -> np.ndarray:
    with Image.open(ROOT / path) as im:
        return np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR, reducing_gap=2.0))


def cache(task: str, df: pd.DataFrame, size: int):
    out = DATA / "cache" / f"{task}_{size}.npy"
    if out.exists():
        arr = np.load(out, mmap_mode="r")
        if arr.shape[0] == len(df):
            print(f"  cache exists: {out.name}")
            return
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.lib.format.open_memmap(out, mode="w+", dtype=np.uint8, shape=(len(df), size, size, 3))
    if df.path.iloc[0].startswith("parquet:"):
        _cache_parquet(df, size, arr)
        arr.flush()
        print(f"  wrote {out.name}  {arr.shape}")
        return
    with ThreadPoolExecutor(max_workers=16) as ex:
        for i, img in enumerate(ex.map(lambda p: _load(p, size), df.path, chunksize=64)):
            arr[i] = img
    arr.flush()
    print(f"  wrote {out.name}  {arr.shape}")


def _cache_parquet(df: pd.DataFrame, size: int, arr):
    """Stream image bytes shard by shard (C-NMC is ~6.5 GB of BMPs, so never load it all at once)."""
    import io
    import pyarrow.parquet as pq
    loc = df.path.str.slice(len("parquet:")).str.rsplit("#", n=1, expand=True)
    where = {(f, int(r)): i for i, (f, r) in enumerate(zip(loc[0], loc[1]))}

    def decode(blob):
        if blob is None:
            return None
        with Image.open(io.BytesIO(blob)) as im:
            return np.asarray(im.convert("RGB").resize((size, size), Image.Resampling.BILINEAR, reducing_gap=2.0))

    with ThreadPoolExecutor(max_workers=16) as ex:
        for f in loc[0].unique():
            row = 0
            for batch in pq.ParquetFile(ROOT / f).iter_batches(columns=["image"], batch_size=256):
                blobs = [d["bytes"] if (f, row + j) in where else None  # skip excluded rows
                         for j, d in enumerate(batch.column("image").to_pylist())]
                for j, img in enumerate(ex.map(decode, blobs)):
                    i = where.get((f, row + j))
                    if i is not None:
                        arr[i] = img
                row += len(blobs)
            print(f"    {f.rsplit('/', 1)[-1]} done", flush=True)


def summarize(task, df, split_col):
    print(f"\n[{task}] {len(df)} images")
    print(pd.crosstab(df.label, df[split_col], margins=True).to_string())
    if "patient" in df:
        print("patients per split:", df.groupby(split_col).patient.nunique().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["pbc", "cnmc"])
    ap.add_argument("--sizes", nargs="+", type=int, default=[224])
    args = ap.parse_args()
    (DATA / "splits").mkdir(parents=True, exist_ok=True)

    for task in args.tasks:
        ensure_raw("PBC" if task == "pbc" else "C-NMC")
        if task == "pbc":
            df, classes, col = scan_pbc(), PBC_CLASSES, "split"
        else:
            df, classes, col = scan_cnmc(), CNMC_CLASSES, "split_patient"
        df.to_csv(DATA / "splits" / f"{task}.csv", index=False)
        (DATA / "splits" / f"{task}_classes.json").write_text(json.dumps(classes))
        summarize(task, df, col)
        for s in args.sizes:
            cache(task, df, s)


if __name__ == "__main__":
    main()
