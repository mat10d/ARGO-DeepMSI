from argo_deepmsi.scorers.nested_linear_probe import NestedLinearProbe
from argo_deepmsi.scorers.registry import get_scorer


def test_nested_probe_contract():
    scorer = get_scorer("nested_linear_probe")
    assert isinstance(scorer, NestedLinearProbe)
    assert scorer.needs_training_on_our_data is True
    assert scorer.resolution == "slide"
    assert scorer.patient_aggregation == "mean"
    assert scorer.primary_score == "p_msih"
