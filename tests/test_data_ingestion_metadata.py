from __future__ import annotations

from pathlib import Path

import pandas as pd

from argo_deepmsi.data_ingestion import load_halo_link_data, load_pathpresenter_data


def _export_row(name: str, record_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Slide ID": [f"id-{name}"],
            "Name": [name],
            "Pathology REDCap ID": [record_id],
            "Cut location": ["OAUTHC"],
            "Stain location": ["OAUTHC"],
        }
    )


def test_pathpresenter_csv_uses_parent_directory_as_site(tmp_path: Path):
    site_dir = tmp_path / "OAUTHC"
    site_dir.mkdir()
    _export_row("slide.svs", "142-1").to_csv(
        site_dir / "pathpresenter_export.csv", index=False
    )
    pd.DataFrame({"unrelated": [1]}).to_csv(site_dir / "other.csv", index=False)

    metadata = load_pathpresenter_data(tmp_path)

    assert metadata[["filename", "redcap_id", "site"]].to_dict("records") == [
        {"filename": "slide.svs", "redcap_id": "142-1", "site": "OAUTHC"}
    ]


def test_pathpresenter_excel_supports_multiple_site_sheets(
    tmp_path: Path, monkeypatch
):
    workbook = tmp_path / "pathpresenter.xlsx"
    workbook.touch()
    monkeypatch.setattr(
        pd,
        "read_excel",
        lambda *_args, **_kwargs: {
            "LASUTH": _export_row("a.svs", "144-1"),
            "LUTH": _export_row("b.svs", "144-2"),
        },
    )

    metadata = load_pathpresenter_data(tmp_path)

    assert metadata["site"].tolist() == ["LASUTH", "LUTH"]
    assert metadata["filename"].tolist() == ["a.svs", "b.svs"]


def test_historical_halo_loader_is_a_compatibility_alias(tmp_path: Path):
    site_dir = tmp_path / "UITH"
    site_dir.mkdir()
    _export_row("legacy.svs", "144-3").to_csv(site_dir / "halo_link_export.csv", index=False)

    assert load_halo_link_data(tmp_path).equals(load_pathpresenter_data(tmp_path))
