"""Write the X1 backend-equivalence slide subset (8 slides spread across every SITE).

Uses the same selection as ``argo compare-backends --select sites --seed 0`` on the slide
table ARGO extracts from (``slide_table_pyramidal.csv``: non-pyramidal/MPP-less slides are
replaced by their ``.pyramidal.tiff``), so both backends read the identical file per slide
and ``compare-backends --slides <subset> --n 8`` re-selects exactly these rows.

Usage:
    uv run --frozen python scripts/backends/x1_select_slides.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from argo_deepmsi.cli import _select_slides

SOURCE = Path("results/data/slide_table_pyramidal.csv")
OUT = Path("results/analysis/backends/x1_slides.csv")


def main() -> None:
    table = pd.read_csv(SOURCE)
    chosen = _select_slides(table, 8, "sites", seed=0)
    chosen = chosen.assign(source_row=chosen.index)
    missing = [path for path in chosen["FILENAME"] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing slides: {missing}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(OUT, index=False)
    print(chosen[["FILENAME", "SITE", "source_row"]].to_string())
    print(f"sites: {sorted(chosen['SITE'].unique())}; wrote {OUT}")


if __name__ == "__main__":
    main()
