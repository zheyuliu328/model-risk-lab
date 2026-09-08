import json

import pytest

from model_risk_lab import report
from model_risk_lab.report import generate, verify_bundle


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
    assert len(verify_bundle(first)) == 8


def test_existing_bundle_is_preserved_when_new_seed_is_requested(tmp_path):
    output = tmp_path / "report"
    generate(output)
    original = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError, match="choose a new directory"):
        generate(output, seed=20260908)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == original


def test_interrupted_generation_does_not_publish_partial_evidence(tmp_path, monkeypatch):
    def fail_save(fig, path):
        report.plt.close(fig)
        raise OSError("simulated chart write failure")

    monkeypatch.setattr(report, "_save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        generate(tmp_path / "failed-report")
    assert list(tmp_path.iterdir()) == []


def test_an_output_created_during_generation_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "report"
    save = report._save

    def competing_writer(fig, path):
        output.mkdir(exist_ok=True)
        (output / "keep.txt").write_text("another run's evidence")
        save(fig, path)

    monkeypatch.setattr(report, "_save", competing_writer)
    with pytest.raises(FileExistsError, match="appeared during generation"):
        generate(output)
    assert {p.name: p.read_text() for p in output.iterdir()} == {
        "keep.txt": "another run's evidence"}
    assert list(tmp_path.iterdir()) == [output]


def test_checksum_check_detects_changed_missing_and_extra_evidence(tmp_path):
    output = tmp_path / "report"
    generate(output)
    chart = output / "credit_validation.svg"
    original = chart.read_bytes()
    chart.write_bytes(original + b"<!-- modified -->")
    with pytest.raises(ValueError, match="checksum mismatch: credit_validation.svg"):
        verify_bundle(output)
    chart.unlink()
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(output)
    chart.write_bytes(original)
    (output / "unexpected.txt").write_text("unrecorded evidence")
    with pytest.raises(ValueError, match="inventory"):
        verify_bundle(output)


def test_existing_empty_directory_and_dangling_symlink_are_preserved(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    link = tmp_path / "symlink"
    link.symlink_to(tmp_path / "absent", target_is_directory=True)
    for output in (empty, link):
        with pytest.raises(FileExistsError):
            generate(output)
    assert empty.is_dir()
    assert link.is_symlink()


@pytest.mark.parametrize("manifest", [
    [], None, {"schema_version": 1, "sha256": list(report.EVIDENCE_FILES)},
    {"schema_version": 1, "sha256": dict.fromkeys(report.EVIDENCE_FILES, 7)},
])
def test_invalid_manifest_types_raise_a_clear_validation_error(tmp_path, manifest):
    for name in report.EVIDENCE_FILES:
        (tmp_path / name).write_text("placeholder evidence")
    (tmp_path / "checksums.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="manifest"):
        verify_bundle(tmp_path)
