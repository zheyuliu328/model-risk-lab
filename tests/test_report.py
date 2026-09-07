import json

from model_risk_lab.report import generate


def test_report_reproduces_inputs_predictions_and_results(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    generate(first)
    generate(second)
    for name in ("synthetic_quarters.csv", "credit_predictions.csv", "fx_bumps.csv", "results.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    result = json.loads((first / "results.json").read_text())
    assert len(result["fx"]) == 270
    assert result["manifest"]["seed"] == 20260907
    assert "REPORT.md" in [p.name for p in first.iterdir()]
