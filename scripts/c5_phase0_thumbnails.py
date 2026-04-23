"""C5 Phase 0d — sample low-P and high-P slide thumbnails for visual QC.

For each of a stratified random sample (N per class), render a figure with:
  panel 1 — full slide thumbnail (with tissue contours)
  panel 2 — tile grid overlay
  title — slide_id, patient, site, n_tiles, Wagner P, MSI label

Goal: ground-truth sanity check that low-P slides are non-tumor (fat, normal
mucosa, lymph node, margin) and high-P slides are clear tumor.

Usage:
    python scripts/c5_phase0_thumbnails.py [--n 20] [--low-threshold 0.10]
                                           [--high-threshold 0.60]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import lazyslide as zs
from wsidata import open_wsi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("c5_thumbs")

OUTDIR = Path("results/analysis/c5_phase0/thumbnails")
OUTDIR.mkdir(parents=True, exist_ok=True)


def _slide_path_from_table(slide_id: str, slide_tbl: pd.DataFrame) -> Path | None:
    """Resolve slide_id to the actual WSI file via the pyramidal table."""
    # slide_id strips the suffix; the table's FILENAME may be the raw .svs or
    # a .pyramidal.tiff. Match by stem.
    match = slide_tbl[slide_tbl["stem"] == slide_id]
    if len(match) == 0:
        return None
    return Path(match["FILENAME"].iloc[0])


def _render(slide_path: Path, meta: dict, out: Path) -> bool:
    try:
        zarr_path = slide_path.with_suffix(".zarr")
        if zarr_path.exists():
            wsi = open_wsi(str(slide_path), store=str(slide_path.parent),
                           attach_thumbnail=False)
        else:
            wsi = open_wsi(str(slide_path), attach_thumbnail=False)
            zs.pp.find_tissues(wsi)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
        try:
            zs.pl.tissue(wsi, ax=axes[0], show_contours=True, show_id=False)
        except Exception:
            axes[0].set_title("tissue failed")
        axes[0].set_title("Tissue")
        try:
            zs.pl.tiles(wsi, ax=axes[1])
        except Exception:
            axes[1].set_title("tiles unavailable")
        axes[1].set_title("Tiles")

        fig.suptitle(
            f"{meta['slide_id']}  |  {meta['patient_id']} ({meta['site']})  "
            f"|  n_tiles={meta['n_tiles']}  |  Wagner P={meta['p_msih']:.3f}  "
            f"|  y={'MSI-H' if meta['y'] == 1 else 'MSS'}",
            fontsize=10,
        )
        fig.tight_layout()
        fig.savefig(out, bbox_inches="tight")
        plt.close(fig)
        return True
    except Exception as e:
        log.warning(f"failed {slide_path.name}: {e}")
        plt.close("all")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="slides per class")
    ap.add_argument("--low-threshold", type=float, default=0.10)
    ap.add_argument("--high-threshold", type=float, default=0.60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ws = pd.read_csv("results/analysis/wagner_zeroshot/slide_scores.csv")
    cl = pd.read_csv("results/data/clinical_table.csv")[["PATIENT", "isMSIH"]]
    cl = cl.rename(columns={"PATIENT": "patient_id"})
    ws = ws.merge(cl, on="patient_id", how="left")

    slide_tbl = pd.read_csv("results/data/slide_table_pyramidal.csv")
    slide_tbl["stem"] = slide_tbl["FILENAME"].apply(
        lambda p: Path(p).name.replace(".pyramidal.tiff", "").replace(".svs", ""))

    rng = np.random.default_rng(args.seed)

    low = ws[ws["p_msih"] < args.low_threshold]
    high = ws[ws["p_msih"] >= args.high_threshold]
    log.info(f"candidate pool: {len(low)} low-P (<{args.low_threshold}), "
             f"{len(high)} high-P (>={args.high_threshold})")

    # Stratified sampling
    def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
        n = min(n, len(df))
        idx = rng.choice(df.index, size=n, replace=False)
        return df.loc[idx].sort_values("p_msih")

    low_s = _sample(low, args.n)
    high_s = _sample(high, args.n)

    samples = {"low": low_s, "high": high_s}
    manifest = []
    for label, df in samples.items():
        out_sub = OUTDIR / label
        out_sub.mkdir(exist_ok=True)
        for _, row in df.iterrows():
            sp = _slide_path_from_table(row["slide_id"], slide_tbl)
            if sp is None or not sp.exists():
                log.warning(f"slide path not found for {row['slide_id']}")
                manifest.append(dict(label=label, **row.to_dict(),
                                     status="missing"))
                continue
            out = out_sub / f"{row['slide_id']}.png"
            ok = _render(sp, row.to_dict(), out)
            manifest.append(dict(label=label, **row.to_dict(),
                                 path=str(sp),
                                 status="ok" if ok else "failed"))
            log.info(f"[{label}] {row['slide_id']}  P={row['p_msih']:.3f}  "
                     f"{'OK' if ok else 'FAIL'}")

    pd.DataFrame(manifest).to_csv(OUTDIR / "manifest.csv", index=False)
    log.info(f"done → {OUTDIR}/")


if __name__ == "__main__":
    main()
