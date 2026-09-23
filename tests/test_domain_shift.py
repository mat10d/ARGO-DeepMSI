"""Synthetic contracts for the domain-shift quantification / mitigation / calibration."""

import numpy as np
import pandas as pd

from argo_deepmsi.eval.domain_shift import (
    _kl,
    dann_nested,
    expected_calibration_error,
    fit_predict_dann,
    frechet_distance,
    kernel_inception_distance,
    nested_calibration,
    permutation_frechet,
)


def _frame(n_pat=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_pat):
        y = int(p % 4 == 0)
        site = ["A", "B", "C"][p % 3]
        for s in range(2):
            rows.append({"slide_id": f"s{p}_{s}", "patient_id": f"P{p}", "site": site, "y": y})
    frame = pd.DataFrame(rows)
    X = rng.normal(size=(len(frame), 8))
    X[:, 0] += 2.0 * frame["y"].to_numpy()
    X[:, 1] += 3.0 * (frame["site"] == "B").to_numpy()
    return frame, X


def test_frechet_zero_for_same_and_grows_with_shift():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(400, 4))
    b = rng.normal(size=(400, 4))
    assert frechet_distance(a, b) < 0.2
    assert frechet_distance(a, b + 2.0) > 10 * frechet_distance(a, b)


def test_permutation_null_flags_real_shift_only():
    frame, X = _frame()
    groups = frame["patient_id"].to_numpy()
    sites = frame["site"].to_numpy()
    shifted = permutation_frechet(X, sites == "B", sites == "A", groups, n_perm=50)
    same = permutation_frechet(X, sites == "C", sites == "A", groups, n_perm=50)
    assert shifted["p"] < 0.05 and shifted["fd_ratio"] > 2
    assert same["fd_ratio"] < shifted["fd_ratio"]


def test_kl_and_kid_basic():
    assert _kl(np.array([1.0, 1.0]), np.array([1.0, 1.0])) < 1e-9
    rng = np.random.default_rng(1)
    a = rng.normal(size=(300, 16))
    assert kernel_inception_distance(a, a + 1.0, n_sub=100, n_rep=3) > abs(
        kernel_inception_distance(a, rng.normal(size=(300, 16)), n_sub=100, n_rep=3)
    )


def test_dann_predicts_probabilities_and_learns_label():
    frame, X = _frame()
    y = frame["y"].to_numpy().astype(float)
    p = fit_predict_dann(X, y, frame["site"].to_numpy(), X, lam=0.3, epochs=100)
    assert p.shape == (len(frame),) and np.all((p >= 0) & (p <= 1))
    assert p[y == 1].mean() > p[y == 0].mean()


def test_dann_nested_emits_oof_for_every_slide():
    frame, X = _frame()
    oof, audit = dann_nested(frame, X, lams=(0.0, 0.3), repeats=1)
    assert len(oof) == len(frame) and oof["p_dann"].notna().all()
    assert set(audit["selected_lam"]) <= {0.0, 0.3}


def test_nested_calibration_methods_and_ece():
    rng = np.random.default_rng(0)
    y = (rng.random(200) < 0.2).astype(int)
    patients = pd.DataFrame(
        {"y": y, "site": np.where(rng.random(200) < 0.5, "A", "B"),
         "score": 0.3 * y + rng.random(200) * 0.5}
    )
    pooled, per_site = nested_calibration(patients, repeats=3)
    assert set(pooled["method"]) == {"uncalibrated", "platt", "isotonic"}
    assert set(per_site["site"]) == {"A", "B"}
    assert expected_calibration_error(y, y.astype(float)) == 0.0


def test_moment_match_maps_source_onto_reference():
    from argo_deepmsi.eval.site_smoothing import MomentAccumulator, moment_match

    rng = np.random.default_rng(0)
    src = rng.normal(3.0, 2.0, size=(2000, 4))
    ref = rng.normal(0.0, 1.0, size=(2000, 4))
    acc = MomentAccumulator(4, full_cov=True)
    acc.add("s", src)
    acc.add("r", ref)
    for mode in ("diag", "coral"):
        out = moment_match(src, acc.moments("s"), acc.moments("r"), mode=mode)
        assert np.allclose(out.mean(0), ref.mean(0), atol=0.1)
        assert np.allclose(out.std(0), ref.std(0), atol=0.1)
    half = moment_match(src, acc.moments("s"), acc.moments("r"), alpha=0.5)
    assert np.allclose(half.mean(0), (src.mean(0) + ref.mean(0)) / 2, atol=0.1)


def test_site_score_normalise_and_invariant_descriptor():
    from argo_deepmsi.eval.site_smoothing import invariant_bag_descriptor, site_score_normalise

    pts = pd.DataFrame({"site": ["A"] * 5 + ["B"] * 5,
                        "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]})
    z = site_score_normalise(pts, "znorm")
    assert abs(z[pts.site == "A"].median()) < 1e-9 and abs(z[pts.site == "B"].median()) < 1e-9
    r = site_score_normalise(pts, "rank")
    assert r.max() == 1.0 and r.min() == 0.2
    X = np.random.default_rng(0).normal(size=(200, 6))
    assert np.allclose(invariant_bag_descriptor(X), invariant_bag_descriptor(3 * X + 5), atol=1e-4)


def test_slide_count_audit_columns(tmp_path, monkeypatch):
    from argo_deepmsi.eval import domain_shift as ds

    frame, _ = _frame()
    frame.to_csv(tmp_path / "cohort.csv", index=False)
    rng = np.random.default_rng(0)
    scores = frame.assign(p_msih=0.3 * frame["y"] + rng.random(len(frame)))
    scores.to_csv(tmp_path / "scores.csv", index=False)
    monkeypatch.setattr(ds, "load_cohort", lambda: frame)
    out = ds.slide_count_audit(tmp_path / "scores.csv")
    assert {"auroc_max_sqrtn", "auroc_mean", "auroc_one_random_slide",
            "auroc_minus_n_slides"} <= set(out.columns)
    assert out.loc[out.site == "ALL", "auroc_mean"].item() > 0.6
