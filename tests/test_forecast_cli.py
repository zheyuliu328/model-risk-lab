"""CLI acceptance: actual published evidence, read-only sources, and no replacement."""

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

from model_risk_lab import forecast_cli


ROOT = Path(__file__).parents[1]
EXPECTED_FILES = {
    "request.json",
    "result.json",
    "candidates.csv",
    "folds.csv",
    "predictions.csv",
    "input_rows.csv",
    "failures.csv",
    "REPORT.md",
    "manifest.json",
}


def run_cli(tmp_path, *arguments):
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment.pop("PYTHONHOME", None)
    return subprocess.run(
        [sys.executable, "-m", "model_risk_lab.forecast_cli", *map(str, arguments)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )


def source_request(tmp_path, constant=False):
    targets = [99, 1, 3, 2, 6, 5, 8, 10, 12, 14]
    request = {
        "schema_version": 1,
        "title": "<script>alert(1)</script> [image](https://example.invalid/track)",
        "rows": [
            {
                "period": f"2020-{index + 1:02d}",
                "target": targets[index],
                "features": {"f1": 5 if constant else index},
            }
            for index in reversed(range(10))
        ],
        "features": [{"id": "f1", "name": "Hand arithmetic", "lag": 1, "release_delay": 0}],
        "horizon": 1,
        "development_end": "2020-07",
        "n_splits": 1,
        "validation_months": 2,
        "min_train": 3,
        "target": {"name": "Hand target", "unit": "index points", "transformation": "none"},
        "source_note": "Invented fixture only.",
    }
    path = tmp_path / "source request.json"
    path.write_text(json.dumps(request, indent=1), encoding="utf-8")
    return path


def bundle(path):
    files = {entry.name: entry.read_bytes() for entry in path.iterdir()}
    assert set(files) == EXPECTED_FILES
    manifest = json.loads(files["manifest.json"])
    assert set(manifest["files_sha256"]) == EXPECTED_FILES - {"manifest.json"}
    for name, digest in manifest["files_sha256"].items():
        assert hashlib.sha256(files[name]).hexdigest() == digest
    result = json.loads(files["result.json"])
    assert result["data_fingerprint"] == manifest["data_fingerprint"]
    assert result["selection_fingerprint"] == manifest["selection_fingerprint"]
    return files, manifest, result


def table(content):
    return list(csv.DictReader(io.StringIO(content.decode("utf-8"))))


def test_cli_example_publishes_complete_development_evidence_and_source_code_hashes(tmp_path):
    destination = tmp_path / "example"
    completed = run_cli(tmp_path, "--example", "--output", destination)
    assert completed.returncode == 0, completed.stderr
    files, manifest, result = bundle(destination)
    assert json.loads(completed.stdout)["selected_on_development"] == "ols-f1-f2"
    assert result["stage"] == "development"
    assert len(table(files["candidates.csv"])) == 17
    assert len(table(files["folds.csv"])) == 51
    assert len(table(files["input_rows.csv"])) == 180
    predictions = table(files["predictions.csv"])
    assert len(predictions) == 36
    assert {row["split"] for row in predictions} == {"development"}
    assert all(row["holdout_mae"] == "" for row in table(files["candidates.csv"]))
    assert b"Holdout remains unscored" in files["REPORT.md"]
    for name, digest in manifest["code_sha256"].items():
        assert (
            hashlib.sha256((ROOT / "src/model_risk_lab" / name).read_bytes()).hexdigest() == digest
        )
    assert manifest["request_source"]["kind"] == "synthetic_example"


def test_cli_request_normalizes_order_and_exports_independent_arithmetic_and_original_hash(
    tmp_path,
):
    source = source_request(tmp_path)
    original = source.read_bytes()
    destination = tmp_path / "published"
    completed = run_cli(tmp_path, "--request", source.name, "--output", destination.name)
    assert completed.returncode == 0, completed.stderr
    files, manifest, result = bundle(destination)
    assert source.read_bytes() == original
    assert manifest["request_source"] == {
        "kind": "file",
        "name": source.name,
        "sha256": hashlib.sha256(original).hexdigest(),
        "bytes": len(original),
    }
    normalized = json.loads(files["request.json"])
    assert normalized["rows"][0]["period"] == "2020-01"
    normalized_bytes = json.dumps(
        normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    assert hashlib.sha256(normalized_bytes).hexdigest() == result["data_fingerprint"]
    candidate = next(row for row in table(files["candidates.csv"]) if row["id"] == "ols-f1")
    assert float(candidate["development_mae"]) == pytest.approx(0.8)
    assert float(candidate["development_rmse"]) == pytest.approx(math.sqrt(1.13))
    fold = next(row for row in table(files["folds.csv"]) if row["model_id"] == "ols-f1")
    assert json.loads(fold["fit"])["coefficients"]["f1"] == pytest.approx(1.4)
    predictions = table(files["predictions.csv"])
    assert [float(row["ols-f1"]) for row in predictions] == pytest.approx([6.5, 7.9])
    assert [float(row["actual"]) for row in predictions] == [5, 8]
    assert b"<script>" not in files["REPORT.md"]
    assert b"&lt;script&gt;" in files["REPORT.md"]
    assert b"[image](https://example.invalid/track)" not in files["REPORT.md"]
    assert str(source.parent).encode() not in files["manifest.json"]


def test_repeated_runs_are_deterministic_but_existing_output_is_never_replaced(tmp_path):
    source = source_request(tmp_path)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    first, second = tmp_path / "first", tmp_path / "second"
    assert run_cli(tmp_path, "--request", source, "--output", first).returncode == 0
    assert run_cli(tmp_path, "--request", source, "--output", second).returncode == 0
    initial, _, _ = bundle(first)
    repeated, _, _ = bundle(second)
    assert initial == repeated
    denied = run_cli(tmp_path, "--example", "--reveal-holdout", "--output", first)
    assert denied.returncode == 2
    assert "already exists" in denied.stderr
    assert {path.name: path.read_bytes() for path in first.iterdir()} == initial
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert not list(tmp_path.glob(".*.staging-*"))


def test_failed_candidates_are_published_with_code_one_and_row_fold_failure_evidence(tmp_path):
    source = source_request(tmp_path, constant=True)
    destination = tmp_path / "failed models"
    completed = run_cli(tmp_path, "--request", source, "--output", destination)
    assert completed.returncode == 1, completed.stderr
    files, _, result = bundle(destination)
    assert result["selected_on_development"] is None
    candidate = next(row for row in table(files["candidates.csv"]) if row["id"] == "ols-f1")
    assert candidate["status"] == "failed"
    assert "constant_feature" in candidate["reason"]
    assert candidate["development_n"] == ""
    failures = table(files["failures.csv"])
    assert {row["scope"] for row in failures} == {"model", "fold", "input"}
    assert any(
        row["scope"] == "input" and row["code"] == "warmup" and row["period"] == "2020-01"
        for row in failures
    )
    assert all(
        row["ols-f1"] == "" and row["baseline-mean"] != ""
        for row in table(files["predictions.csv"])
    )
    assert b"No eligible OLS candidate" in files["REPORT.md"]
    assert len(table(files["input_rows.csv"])) == 10


def test_explicit_reveal_only_adds_holdout_evaluation_without_reselection(tmp_path):
    source = source_request(tmp_path)
    hidden, revealed = tmp_path / "hidden", tmp_path / "revealed"
    assert run_cli(tmp_path, "--request", source, "--output", hidden).returncode == 0
    assert (
        run_cli(tmp_path, "--request", source, "--reveal-holdout", "--output", revealed).returncode
        == 0
    )
    before_files, before_manifest, before = bundle(hidden)
    files, manifest, result = bundle(revealed)
    assert result["stage"] == "holdout"
    assert result["selected_on_development"] == before["selected_on_development"]
    assert manifest["selection_fingerprint"] == before_manifest["selection_fingerprint"]
    assert manifest["data_fingerprint"] == before_manifest["data_fingerprint"]
    assert result["folds"] == before["folds"]
    assert files["request.json"] == before_files["request.json"]
    held = [row for row in table(files["predictions.csv"]) if row["split"] == "holdout"]
    assert len(held) == 3
    assert [float(row["baseline-persistence"]) for row in held] == [8, 10, 12]
    assert b"## Revealed holdout" in files["REPORT.md"]


def test_revealed_empty_common_holdout_publishes_coverage_and_returns_one(tmp_path):
    source = source_request(tmp_path)
    request = json.loads(source.read_text())
    for row in request["rows"]:
        if row["period"] >= "2020-07":
            row["features"]["f1"] = None
    source.write_text(json.dumps(request), encoding="utf-8")
    destination = tmp_path / "no holdout sample"
    completed = run_cli(tmp_path, "--request", source, "--reveal-holdout", "--output", destination)
    assert completed.returncode == 1, completed.stderr
    files, _, result = bundle(destination)
    assert result["selected_on_development"] == "ols-f1"
    assert result["coverage"]["holdout"] == {"raw": 3, "usable": 0, "excluded": 3}
    failures = [row for row in table(files["failures.csv"]) if row["scope"] == "holdout"]
    assert len(failures) == 3
    assert {row["reason"] for row in failures} == {"no_common_holdout_rows"}
    assert all(row["split"] == "development" for row in table(files["predictions.csv"]))


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "malformed",
        "duplicate_json_key",
        "missing_target",
        "nonstandard_number",
        "oversized",
    ],
)
def test_invalid_request_exits_two_without_publishing_or_mutating_source(tmp_path, case):
    source = source_request(tmp_path)
    if case == "missing":
        source = tmp_path / "absent.json"
    elif case == "malformed":
        source.write_text('{"schema_version":', encoding="utf-8")
    elif case == "duplicate_json_key":
        source.write_text('{"schema_version":1,"schema_version":2}', encoding="utf-8")
    elif case == "missing_target":
        request = json.loads(source.read_text())
        request["rows"][0]["target"] = None
        source.write_text(json.dumps(request), encoding="utf-8")
    elif case == "nonstandard_number":
        source.write_text('{"schema_version":NaN}', encoding="utf-8")
    elif case == "oversized":
        with source.open("wb") as stream:
            stream.seek(forecast_cli.MAX_REQUEST_BYTES)
            stream.write(b" ")
    original = source.read_bytes() if source.exists() else None
    output = tmp_path / "should not appear"
    completed = run_cli(tmp_path, "--request", source, "--output", output)
    assert completed.returncode == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".*.staging-*"))
    assert "Traceback" not in completed.stderr
    if original is not None:
        assert source.read_bytes() == original


@pytest.mark.parametrize("populated", [False, True])
def test_racing_output_creation_cannot_replace_an_empty_or_populated_directory(
    tmp_path, monkeypatch, populated
):
    source = source_request(tmp_path)
    original = source.read_bytes()
    destination = tmp_path / "raced"
    publish = forecast_cli._publish_exclusive

    def competing_publication(staging, output):
        output.mkdir()
        if populated:
            (output / "existing.txt").write_text(
                "Keep this existing user content", encoding="utf-8"
            )
        publish(staging, output)

    monkeypatch.setattr(forecast_cli, "_publish_exclusive", competing_publication)
    assert forecast_cli.main(["--request", str(source), "--output", str(destination)]) == 2
    if populated:
        assert (destination / "existing.txt").read_text() == "Keep this existing user content"
    assert list(destination.iterdir()) == ([destination / "existing.txt"] if populated else [])
    assert not list(tmp_path.glob(".*.staging-*"))
    assert source.read_bytes() == original


def test_failed_publication_cleans_only_its_own_staging_directory(tmp_path, monkeypatch):
    source = source_request(tmp_path)
    existing = tmp_path / "existing user file"
    existing.write_bytes(b"User data is retained")

    def unavailable(staging, output):
        raise OSError("Publication unavailable")

    monkeypatch.setattr(forecast_cli, "_publish_exclusive", unavailable)
    destination = tmp_path / "unpublished"
    assert forecast_cli.main(["--request", str(source), "--output", str(destination)]) == 2
    assert not destination.exists()
    assert not list(tmp_path.glob(".*.staging-*"))
    assert existing.read_bytes() == b"User data is retained"


def test_csv_export_neutralizes_formulas_but_preserves_actual_negative_numbers():
    exported = forecast_cli._csv(
        ["text", "actual"],
        [
            {"text": '\t=HYPERLINK("https://example.invalid")', "actual": -2.5},
            {"text": "@SUM(1,2)", "actual": 0},
        ],
    )
    rows = table(exported)
    assert rows[0]["text"].startswith("'\t=")
    assert rows[1]["text"].startswith("'@")
    assert rows[0]["actual"] == "-2.5"
