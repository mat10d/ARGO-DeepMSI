"""Build a patient-level JHU/CRC/pathology REDCap crosswalk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _present(value: object) -> bool:
    return value is not None and str(value).strip() not in {"", "nan", "NaN"}


def _normalise(value: object) -> str:
    return "".join(ch for ch in str(value or "").strip().lower() if ch.isalnum()).lstrip("0")


def build_crosswalk(
    crc_records: list[dict[str, object]], pathology_records: list[dict[str, object]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    slide_fields = ["slide_name"] + [f"slide_name_{i}" for i in range(1, 51)]

    pathology_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in pathology_records:
        for key_name, value in (
            ("crc_record_id", record.get("crc_redcap_number")),
            ("hospital_number", record.get("hospital_number")),
        ):
            key = (key_name, _normalise(value))
            if key[1]:
                pathology_by_key.setdefault(key, []).append(record)

    for crc in (r for r in crc_records if str(r.get("on_jhu_study", "")) == "1"):
        crc_id = str(crc.get("research_number") or "").strip()
        hospital = str(crc.get("hospital_no") or "").strip()
        candidates: list[tuple[str, dict[str, object]]] = []
        for key_name, value in (("crc_record_id", crc_id), ("hospital_number", hospital)):
            for pathology in pathology_by_key.get((key_name, _normalise(value)), []):
                candidates.append((key_name, pathology))
        # Preserve one row per CRC/pathology relationship, removing duplicate
        # matches where both identifiers agree.
        unique: dict[str, tuple[str, dict[str, object]]] = {}
        for method, pathology in candidates:
            unique[str(pathology.get("record_id"))] = (method, pathology)
        if not unique:
            unique[""] = ("unmatched", {})

        for method, record in unique.values():
            slides: list[str] = []
            for field in slide_fields:
                value = record.get(field)
                if _present(value) and str(value).strip() not in slides:
                    slides.append(str(value).strip())

            pathology_id = record.get("record_id")
            row = {
                "crc_record_id": crc_id,
                "r01_record_id": str(record.get("r01_record_id") or "").strip(),
                "pathology_record_id": str(pathology_id).strip() if _present(pathology_id) else "",
                "match_method": method,
                "crc_hospital_no": hospital,
                "pathology_hospital_number": str(record.get("hospital_number") or "").strip(),
                "slide_names": "; ".join(slides),
                "slide_count_redcap": record.get("slide_count", ""),
                "slide_found": record.get("slide_found", ""),
                "slide_pathpresenter": record.get("slide_pathpresenter", ""),
                "on_jhu_study": "Yes",
                "in_crc": True,
                "in_jhu_r01": _present(record.get("r01_record_id")),
                "has_pathology_record": _present(pathology_id),
                "has_slide_name": bool(slides),
            }
            rows.append(row)

    crosswalk = pd.DataFrame(rows).sort_values(
        ["in_jhu_r01", "in_crc", "pathology_record_id"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    # Counts are by distinct identifier, while the overlap table is by
    # pathology REDCap row.  This makes duplicate CRC/R01 links visible.
    summary = pd.DataFrame(
        [
            {"measure": "CRC records on JHU study", "count": crosswalk["crc_record_id"].nunique()},
            {"measure": "JHU CRC records matched to pathology", "count": int((crosswalk.match_method != "unmatched").sum())},
            {"measure": "JHU CRC records with direct CRC-ID match", "count": int((crosswalk.match_method == "crc_record_id").sum())},
            {"measure": "JHU CRC records with hospital-number match", "count": int((crosswalk.match_method == "hospital_number").sum())},
            {"measure": "JHU CRC records without pathology match", "count": int((crosswalk.match_method == "unmatched").sum())},
            {"measure": "Matched pathology records with R01 ID", "count": int(crosswalk.loc[crosswalk.match_method != "unmatched", "in_jhu_r01"].sum())},
            {"measure": "Records with at least one slide name", "count": int(crosswalk.has_slide_name.sum())},
        ]
    )
    combination = (
        crosswalk.assign(
            membership=crosswalk.apply(
                lambda r: "+".join(
                    x
                    for x, flag in (("JHU/R01", r.in_jhu_r01), ("CRC", r.in_crc), ("Pathology", r.has_pathology_record))
                    if flag
                ),
                axis=1,
            )
        )
        .groupby("membership", as_index=False)
        .size()
        .rename(columns={"size": "pathology_record_rows"})
    )
    summary = pd.concat(
        [summary, pd.DataFrame([{"measure": "", "count": ""}]), combination.rename(columns={"membership": "measure", "pathology_record_rows": "count"})],
        ignore_index=True,
    )
    return crosswalk, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path, help="CRC REDCap JSON export")
    parser.add_argument("pathology_json", type=Path, help="Pathology REDCap JSON export")
    parser.add_argument("output_xlsx", type=Path)
    args = parser.parse_args()

    crc_records = json.loads(args.input_json.read_text())
    pathology_records = json.loads(args.pathology_json.read_text())
    crosswalk, summary = build_crosswalk(crc_records, pathology_records)
    args.output_xlsx.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output_xlsx, engine="openpyxl") as writer:
        crosswalk.to_excel(writer, sheet_name="crosswalk", index=False)
        summary.to_excel(writer, sheet_name="counts", index=False)
    crosswalk.to_csv(args.output_xlsx.with_suffix(".csv"), index=False)
    summary.to_csv(args.output_xlsx.with_name(args.output_xlsx.stem + "_counts.csv"), index=False)
    print(f"wrote {len(crosswalk)} rows to {args.output_xlsx}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
