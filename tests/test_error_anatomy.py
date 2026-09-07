import numpy as np
import pandas as pd

from argo_deepmsi.eval.error_anatomy import (
    assign_site_structured,
    attribute_causes,
    build_error_ledger,
    paired_scan_analysis,
    pareto,
)


def _tiny_inputs():
    # P1: two MSS slides, low scores -> TN. P2: one MSI-H slide, low score -> FN (confident miss).
    scores = pd.DataFrame({
        "slide_id": ["s1", "s2", "s3"],
        "patient_id": ["P1", "P1", "P2"],
        "site": ["UITH", "UITH", "OAUTHC"],
        "y": [0, 0, 1],
        "p_msih": [0.10, 0.20, 0.05],
        "stain_location": ["UITH", "UITH", "OAUTHC"],
    })
    cohort = pd.DataFrame({
        "slide_id": ["s1", "s2", "s3"],
        "patient_id": ["P1", "P1", "P2"],
        "processing_site": ["UITH", "UITH", "OAUTHC"],
        "n_tiles": [100, 100, 80],
        "tumor_fraction": [0.5, 0.5, 0.4],
        "artifact_fraction": [0.1, 0.1, 0.2],
        "label_source": ["prospective_cmo", "prospective_cmo", "prospective_cmo"],
        "label_certainty": ["definite", "definite", "definite"],
        "cmo_msi_score": [1.0, 1.0, 30.0],
        "msi_status_mmr": [np.nan, np.nan, np.nan],
        "cmo_msi_status": ["MSS", "MSS", "MSI-H"],
    })
    clinical = pd.DataFrame({
        "PATIENT": ["P1", "P2"],
        "msi_method": ["PCR", "PCR"],
        "slide_staining_site": ["UITH", "OAUTHC"],
        "slide_imaging_site": ["UITH", "OAUTHC"],
    })
    return scores, cohort, clinical


def test_build_error_ledger_aggregation_and_classes():
    scores, cohort, clinical = _tiny_inputs()
    # inject a threshold so all four error classes are deterministic
    ledger, detail = build_error_ledger(scores, cohort, clinical, threshold=0.5)
    p1 = ledger.set_index("patient_id").loc["P1"]
    # max/sqrt(n): max(0.10,0.20)/sqrt(2) = 0.1414 < 0.5, y=0 -> TN
    assert abs(p1["wagner_patient_score"] - 0.20 / np.sqrt(2)) < 1e-9
    assert p1["n_slides"] == 2
    assert p1["error_class"] == "TN"
    p2 = ledger.set_index("patient_id").loc["P2"]
    # score 0.05 < 0.5, y=1 -> FN (confident: residual < -CONFIDENT_MARGIN)
    assert p2["error_class"] == "FN"
    assert bool(p2["confident_wrong"]) is True
    assert len(detail) == 3




def _err_row(**kw):
    base = dict(error_class="FN", confident_wrong=True, label_certainty="definite",
                mmr_cmo_concordance=False, cmo_msi_score=30.0, tumor_fraction=0.5,
                artifact_fraction=0.1, rank_residual=-0.3, site="OAUTHC")
    base.update(kw)
    return base


def test_attribution_priority_and_levers():
    df = pd.DataFrame([
        _err_row(label_certainty="indeterminate_as_mss"),          # -> label-suspect
        _err_row(tumor_fraction=0.0),                              # -> low-tumor-content
        _err_row(artifact_fraction=0.9),                          # -> high-artifact
        _err_row(confident_wrong=False, rank_residual=-0.02),     # -> borderline-score
        dict(error_class="TN", confident_wrong=False),            # -> correct
    ])
    out = attribute_causes(df)
    assert list(out["attributed_cause"]) == [
        "label-suspect", "low-tumor-content", "high-artifact", "borderline-score", "correct"]
    assert out.loc[0, "circumvention_lever"].startswith("re-adjudicat")
    assert out.loc[4, "circumvention_lever"] == ""




def test_site_structured_and_pareto():
    df = pd.DataFrame({
        "attributed_cause": ["unexplained", "unexplained", "correct", "correct"],
        "error_class": ["FN", "FP", "TN", "TP"],
        "site": ["OAUTHC", "OAUTHC", "UITH", "UITH"],
        "circumvention_lever": ["", "", "", ""],
    })
    out = assign_site_structured(df)
    assert set(out.loc[out["site"] == "OAUTHC", "attributed_cause"]) == {"site-structured"}
    p = pareto(out)
    assert p.loc[p["attributed_cause"] == "site-structured", "count"].iloc[0] == 2


def test_enrichment_gating_drops_ubiquitous_covariate():
    # high artifact present in ALL rows (errors AND correct) -> not enriched -> not attributed
    err = [dict(error_class="FP", confident_wrong=True, label_certainty="definite",
                mmr_cmo_concordance=False, cmo_msi_score=1.0, tumor_fraction=0.5,
                artifact_fraction=0.9) for _ in range(4)]
    ok = [dict(error_class="TN", confident_wrong=False, label_certainty="definite",
               mmr_cmo_concordance=False, cmo_msi_score=1.0, tumor_fraction=0.5,
               artifact_fraction=0.9) for _ in range(4)]
    out = attribute_causes(pd.DataFrame(err + ok))
    # 100% artifact in both groups -> enrichment 1.0 < 1.2 -> confident FPs become "unexplained"
    assert set(out.loc[out.error_class == "FP", "attributed_cause"]) == {"unexplained"}
    assert "high-artifact" not in out.attrs["enriched_causes"]




def test_paired_scan_pairs_only_retrospective_dual():
    detail = pd.DataFrame({
        "slide_id": ["a", "b", "c", "d"],
        "patient_id": ["R1", "R1", "R2", "S1"],
        "p_msih": [0.2, 0.8, 0.3, 0.4],
        "processing_site": ["retrospective_msk", "retrospective_oau",
                            "retrospective_msk", "OAUTHC"],
        "stain_location": ["MSK", "OAU", "MSK", "OAUTHC"],
    })
    paired, stats = paired_scan_analysis(detail)
    # R1 has both msk+oau -> paired; R2 only msk; S1 not retrospective -> excluded
    assert list(paired["patient_id"]) == ["R1"]
    assert abs(paired.loc[0, "delta_p"] - (0.8 - 0.2)) < 1e-9
    assert stats["n_paired"] == 1


def test_attention_mass_on_tumor():
    from argo_deepmsi.eval.error_anatomy import attention_mass_on_tumor
    attn = np.array([0.1, 0.6, 0.2, 0.1])   # sums to 1
    tumor_idx = np.array([1, 2])            # tiles 1 and 2 are tumor
    assert abs(attention_mass_on_tumor(attn, tumor_idx, n_tiles=4) - 0.8) < 1e-9


def test_synthesis_percentages_sum_to_100():
    from argo_deepmsi.eval.error_anatomy import synthesize
    ledger = pd.DataFrame({
        "attributed_cause": ["label-suspect", "label-suspect", "high-artifact", "correct"],
        "circumvention_lever": ["x", "x", "y", ""],
    })
    s = synthesize(ledger, b_stats={"flip_rate": 0.3})
    assert abs(s["pct_of_errors"].sum() - 100.0) < 1e-6
    assert set(s["attributed_cause"]) == {"label-suspect", "high-artifact"}
