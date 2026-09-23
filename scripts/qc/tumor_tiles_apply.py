"""Q3-tumor-filter APPLY (harvest step, CPU-only).

Applies the smoke-off winner (CTransPath NCT-CRC-HE 9-class head) to every
in-clean-set slide and writes per-slide tumor-tile indices:

    results/data/tumor_tiles/<slide_id>.npy   int array of TUM tile indices
    results/data/tumor_tiles/tumor_fraction.csv
        slide_id, n_tiles, n_tumor_tiles, tumor_fraction, status

Reads the cached ``ctranspath_tiles`` features from each slide's zarr (no GPU,
no re-extraction). The reduce into cohort_clean.csv + the tumor-fraction floor
drop happen in ``argo_deepmsi.eval.cohort`` (``--tumor-tiles-dir``), so the
floor lives in one place (the manifest).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import zarr
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("q3_apply")

HEAD_PATH = Path("results/analysis/c5_phase1c/tissue_head_ctranspath_nonorm.joblib")


def _load_X(zarr_path: Path) -> np.ndarray | None:
    try:
        return np.asarray(ad.read_zarr(str(zarr_path / "tables" / "ctranspath_tiles")).X)
    except Exception:
        try:
            z = zarr.open_group(str(zarr_path / "tables" / "ctranspath_tiles"), mode="r")
            return np.asarray(z["X"])
        except Exception as e:
            log.warning(f"ctranspath read failed for {zarr_path.name}: {e}")
            return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", default="results/data/cohort_clean.csv", type=Path)
    p.add_argument("--slide-table", default="results/data/slide_table_pyramidal.csv", type=Path)
    p.add_argument("--outdir", default="results/data/tumor_tiles", type=Path)
    a = p.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)
    head = joblib.load(HEAD_PATH)
    clf, labels = head["clf"], head["labels"]
    tum_id = labels.index("TUM")

    coh = pd.read_csv(a.cohort)
    coh = coh[coh["in_clean_set"] == 1]
    st = pd.read_csv(a.slide_table)
    st["slide_id"] = st["FILENAME"].apply(lambda pth: Path(pth).stem)
    src = st[["slide_id", "FILENAME"]].drop_duplicates("slide_id")
    df = coh[["slide_id"]].merge(src, on="slide_id", how="left")

    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df)):
        sid = r["slide_id"]
        zarr_path = Path(r["FILENAME"]).with_suffix(".zarr")
        if not zarr_path.exists():
            rows.append(dict(slide_id=sid, status="no_zarr"))
            continue
        X = _load_X(zarr_path)
        if X is None or X.size == 0:
            rows.append(dict(slide_id=sid, status="no_features"))
            continue
        yhat = clf.predict_proba(X).argmax(axis=1)
        tum_idx = np.where(yhat == tum_id)[0].astype(np.int32)
        np.save(a.outdir / f"{sid}.npy", tum_idx)
        rows.append(dict(
            slide_id=sid, n_tiles=int(len(yhat)),
            n_tumor_tiles=int(len(tum_idx)),
            tumor_fraction=float(len(tum_idx) / len(yhat)) if len(yhat) else 0.0,
            status="ok",
        ))

    out = pd.DataFrame(rows)
    out.to_csv(a.outdir / "tumor_fraction.csv", index=False)
    ok = out[out["status"] == "ok"]
    log.info(f"applied to {len(ok)}/{len(out)} slides; "
             f"median tumor_fraction={ok['tumor_fraction'].median():.4f}")


if __name__ == "__main__":
    main()
