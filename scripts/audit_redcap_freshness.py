"""Privacy-safe, read-only comparison of live REDCap with local clinical snapshots.

The script never writes or prints record identifiers or raw clinical values. It
reports population counts and per-field change counts, with a separate audit for
the patients represented in the current slide benchmark.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from argo_deepmsi.data_ingestion import create_clinical_table, fetch_redcap_data

AUDIT_FIELDS = (
    "isMSIH",
    "cmo_msi_status",
    "cmo_msi_score",
    "msi_method",
    "msi_status_mmr",
    "mlh1",
    "msh2",
    "msh6",
    "pms2",
    "sample_type",
    "tissue_processing_site",
    "slide_staining_site",
    "slide_imaging_site",
)


def _text(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip()


def stable_patient_key(frame: pd.DataFrame) -> pd.Series:
    """Stable REDCap identity independent of sequential ``PATIENT`` numbering."""
    batch = _text(frame["batch_number"])
    record = _text(frame["record_id"])
    crc = _text(frame.get("crc_redcap_number", pd.Series("", index=frame.index)))
    return pd.Series(
        [
            f"retro:{crc_id}" if batch_id in {"1", "2"} else f"pros:{record_id}"
            for batch_id, crc_id, record_id in zip(batch, crc, record)
        ],
        index=frame.index,
    )


def _normalise_field(values: pd.Series, field: str) -> pd.Series:
    if field == "cmo_msi_score":
        numeric = pd.to_numeric(values, errors="coerce")
        return numeric.map(lambda value: "" if pd.isna(value) else f"{value:.12g}")
    return _text(values)


def _field_delta(local: pd.DataFrame, live: pd.DataFrame, keys: set[str]) -> dict[str, dict]:
    left = local[local["_key"].isin(keys)].set_index("_key")
    right = live[live["_key"].isin(keys)].set_index("_key")
    common = left.index.intersection(right.index)
    output = {}
    for field in AUDIT_FIELDS:
        if field not in left and field not in right:
            continue
        local_values = _normalise_field(
            left.reindex(common).get(field, pd.Series("", index=common)), field
        )
        live_values = _normalise_field(
            right.reindex(common).get(field, pd.Series("", index=common)), field
        )
        changed = local_values.ne(live_values)
        output[field] = {
            "local_populated": int(local_values.ne("").sum()),
            "live_populated": int(live_values.ne("").sum()),
            "changed": int(changed.sum()),
            "newly_populated": int((local_values.eq("") & live_values.ne("")).sum()),
            "cleared": int((local_values.ne("") & live_values.eq("")).sum()),
        }
    return output


def audit(
    clinical_csv: Path,
    clinical_full_csv: Path,
    slide_table_csv: Path,
) -> dict:
    live_raw = fetch_redcap_data()
    live, _ = create_clinical_table(live_raw)
    local = pd.read_csv(clinical_csv, dtype=str)
    local_full = pd.read_csv(clinical_full_csv, dtype=str)
    slides = pd.read_csv(slide_table_csv, dtype=str)
    for frame in (live, local, local_full):
        frame["_key"] = stable_patient_key(frame)

    benchmark_keys = set(local.loc[local["PATIENT"].isin(slides["PATIENT"]), "_key"])
    full_keys = set(local_full["_key"])
    live_keys = set(live["_key"])
    benchmark_fields = _field_delta(local, live, benchmark_keys)
    full_fields = _field_delta(local_full, live, full_keys)
    return {
        "audit_time": datetime.now().astimezone().isoformat(),
        "local_snapshot": {
            "clinical_mtime": datetime.fromtimestamp(
                clinical_csv.stat().st_mtime
            ).astimezone().isoformat(),
            "benchmark_patients": len(benchmark_keys),
            "full_clinical_patients": len(full_keys),
            "slides": int(len(slides)),
        },
        "live_redcap": {
            "raw_rows": int(len(live_raw)),
            "unique_patients": len(live_keys),
        },
        "identity_delta": {
            "new_live_patients": len(live_keys - full_keys),
            "local_patients_missing_live": len(full_keys - live_keys),
            "benchmark_patients_missing_live": len(benchmark_keys - live_keys),
        },
        "benchmark_field_delta": benchmark_fields,
        "full_clinical_field_delta": full_fields,
        "benchmark_labels_current": (
            not benchmark_keys - live_keys
            and benchmark_fields.get("isMSIH", {}).get("changed", 0) == 0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clinical", type=Path, default=Path("results/data/clinical_table.csv"))
    parser.add_argument(
        "--clinical-full", type=Path, default=Path("results/data/clinical_table_full.csv")
    )
    parser.add_argument(
        "--slide-table", type=Path, default=Path("results/data/slide_table_pyramidal.csv")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path for the aggregate audit JSON (never contains record IDs)",
    )
    args = parser.parse_args()
    report = audit(args.clinical, args.clinical_full, args.slide_table)
    rendered = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered)
    print(rendered)
    return 0 if report["benchmark_labels_current"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
