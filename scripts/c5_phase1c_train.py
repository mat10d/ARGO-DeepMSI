"""C5 Phase 1c (step 1) — train NCT-CRC-HE-100K tissue classifier on
CTransPath features.

Downloads NCT-CRC-HE-100K train (100K tiles, 9 classes) + CRC-VAL-HE-7K
holdout from HuggingFace (DykeF/NCTCRCHE100K), extracts CTransPath features
via LazySlide's ImageModel wrapper, trains a multinomial logistic regression
tissue classifier, evaluates on the holdout, and persists:

    results/analysis/c5_phase1c/
        nct_crc_features_train.npz      (X[100000, 768], y[100000], label_map)
        nct_crc_features_val.npz        (X[~7180, 768], y[...], label_map)
        tissue_head_ctranspath.joblib   sklearn LogisticRegression
        train_report.txt                holdout classification report

The 9 class labels follow the original Kather naming:
    ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM
"""

from __future__ import annotations

import json
import logging
import os
import tarfile
from pathlib import Path

import joblib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lazyslide.tl._features import load_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("c5_1c_train")

OUTDIR = Path("results/analysis/c5_phase1c")
OUTDIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = Path(".cache/nct_crc")
DATA_ROOT.mkdir(parents=True, exist_ok=True)

HF_REPO = "DykeF/NCTCRCHE100K"
# Default: NONORM (raw H&E, matches CTransPath's pretraining distribution and
# our cohort's slides, which were not stain-normalized before extraction).
# Override with ARGO_NCT_VARIANT=norm to use the Macenko-normalized variant.
_VARIANT = os.environ.get("ARGO_NCT_VARIANT", "nonorm").lower()
TRAIN_TAR = ("NCT-CRC-HE-100K-NONORM.tar.gz" if _VARIANT == "nonorm"
             else "NCT-CRC-HE-100K.tar.gz")
VAL_TAR = "CRC-VAL-HE-7K.tar.gz"
VARIANT_SUFFIX = f"_{_VARIANT}"

LABELS = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]
LABEL_TO_ID = {l: i for i, l in enumerate(LABELS)}


def _download_and_extract(fname: str) -> Path:
    """Download the tar.gz from HF and extract it under DATA_ROOT."""
    extracted_stem = fname.replace(".tar.gz", "")
    target_dir = DATA_ROOT / extracted_stem
    if target_dir.exists() and any(target_dir.iterdir()):
        log.info(f"already extracted: {target_dir}")
        return target_dir

    log.info(f"downloading {fname} from {HF_REPO}...")
    tar_path = hf_hub_download(repo_id=HF_REPO, filename=fname,
                               repo_type="dataset", cache_dir=str(DATA_ROOT))
    log.info(f"extracting {tar_path} → {DATA_ROOT}/...")
    with tarfile.open(tar_path) as tf:
        tf.extractall(path=DATA_ROOT, filter="data")
    if not target_dir.exists():
        candidates = [p for p in DATA_ROOT.iterdir()
                      if p.is_dir() and p.name.startswith(extracted_stem)]
        if candidates:
            target_dir = candidates[0]
    log.info(f"extracted: {target_dir}")
    return target_dir


class TileDataset(Dataset):
    """Labeled 224x224 tile dataset indexed from a class-subdir layout."""

    def __init__(self, root: Path, transform):
        self.transform = transform
        self.items: list[tuple[Path, int]] = []

        # Layout A: class subdirectories (root/ADI/ADI-XXXX.tif, ...)
        # Layout B: flat directory with filename prefix (root/ADI-XXXX.png, ...)
        label_dirs = [root / lab for lab in LABELS if (root / lab).is_dir()]
        if len(label_dirs) >= 6:
            for lab in LABELS:
                lab_dir = root / lab
                if not lab_dir.exists():
                    continue
                for p in sorted(lab_dir.iterdir()):
                    if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg"):
                        self.items.append((p, LABEL_TO_ID[lab]))
        else:
            # Flat: parse first 3-4 chars of stem up to '-' as the class
            exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
            with os.scandir(root) as it:
                for entry in it:
                    if not entry.is_file():
                        continue
                    name = entry.name
                    if not name.lower().endswith(exts):
                        continue
                    lab = name.split("-", 1)[0]
                    if lab in LABEL_TO_ID:
                        self.items.append((Path(entry.path), LABEL_TO_ID[lab]))
        log.info(f"{root} → {len(self.items)} tiles")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, y


@torch.no_grad()
def extract_features(model, ds: TileDataset, device: str,
                     batch_size: int = 128, num_workers: int = 6):
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    feats = np.empty((len(ds), model.encode_dim), dtype=np.float32)
    ys = np.empty(len(ds), dtype=np.int64)
    i = 0
    for x, y in tqdm(loader, desc="extract"):
        x = x.to(device, non_blocking=True)
        out = model.encode_image(x)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy().astype(np.float32)
        feats[i:i + len(x)] = out
        ys[i:i + len(x)] = y.numpy()
        i += len(x)
    return feats[:i], ys[:i]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device: {device}")

    train_dir = _download_and_extract(TRAIN_TAR)
    val_dir = _download_and_extract(VAL_TAR)

    log.info("loading ctranspath via LazySlide...")
    model, _ = load_models("ctranspath")
    model.to(device)
    model.model.eval()

    transform = model.get_transform()

    train_cache = OUTDIR / f"nct_crc_features_train{VARIANT_SUFFIX}.npz"
    val_cache = OUTDIR / f"nct_crc_features_val{VARIANT_SUFFIX}.npz"

    if train_cache.exists():
        z = np.load(train_cache)
        X_tr, y_tr = z["X"], z["y"]
        log.info(f"loaded cached train: X={X_tr.shape}")
    else:
        ds_tr = TileDataset(train_dir, transform)
        X_tr, y_tr = extract_features(model, ds_tr, device)
        np.savez(train_cache, X=X_tr, y=y_tr,
                 labels=np.array(LABELS, dtype=object))

    if val_cache.exists():
        z = np.load(val_cache)
        X_va, y_va = z["X"], z["y"]
    else:
        ds_va = TileDataset(val_dir, transform)
        X_va, y_va = extract_features(model, ds_va, device)
        np.savez(val_cache, X=X_va, y=y_va,
                 labels=np.array(LABELS, dtype=object))
    log.info(f"train: X={X_tr.shape} y={np.bincount(y_tr)}")
    log.info(f"val:   X={X_va.shape} y={np.bincount(y_va)}")

    log.info("fitting logistic regression head (9-class, L2)...")
    clf = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, solver="lbfgs")
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_va)
    report = classification_report(y_va, y_pred, target_names=LABELS, digits=3)
    cm = confusion_matrix(y_va, y_pred)
    log.info("\n" + report)
    log.info(f"\nconfusion matrix (rows=true, cols=pred):\n{cm}")

    # Save
    head_path = OUTDIR / f"tissue_head_ctranspath{VARIANT_SUFFIX}.joblib"
    joblib.dump(dict(clf=clf, labels=LABELS, backbone="ctranspath",
                     variant=_VARIANT), head_path)
    with open(OUTDIR / f"train_report{VARIANT_SUFFIX}.txt", "w") as f:
        f.write(f"backbone: ctranspath (variant={_VARIANT})\n")
        f.write(f"train: {X_tr.shape}, val: {X_va.shape}\n\n")
        f.write(report + "\n\n")
        f.write("confusion matrix:\n")
        f.write(str(cm))

    log.info(f"done → {head_path}")


if __name__ == "__main__":
    main()
