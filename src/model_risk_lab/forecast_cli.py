"""Offline command line and reproducible evidence export for monthly OLS."""

from __future__ import annotations

import argparse
import csv
import ctypes
import errno
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

from . import __version__, forecast

MAX_REQUEST_BYTES = 10 * 1024 * 1024


def _json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _csv_cell(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        value = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
    if isinstance(value, str) and (
        value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@"))
        or value.startswith(("\t", "\r", "\n"))
    ):
        return "'" + value
    return value


def _csv(headers: list[str], rows: list[dict]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n", extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(row.get(key)) for key in headers})
    return stream.getvalue().encode("utf-8")


def _md(value: object) -> str:
    value = html.escape(str(value), quote=True)
    value = value.replace("\\", "\\\\")
    for character in "`*_{}[]()#+-.!|":
        value = value.replace(character, "\\" + character)
    return value.replace("\r", " ").replace("\n", " ")


def _display(value: float | int | None) -> str:
    return "—" if value is None else f"{value:.8g}"


def _report(result: dict, source: dict) -> bytes:
    protocol = result["protocol"]
    selected = result["selected_on_development"]
    coverage = result["coverage"]
    lines = [
        f"# {_md(result['title'])}",
        "",
        f"Stage: **{result['stage']}**. Development selection: **{selected or 'No eligible OLS candidate'}**.",
        "",
        "This is an OLS candidate screen using caller-supplied monthly observations. It is not model approval or evidence of causal effects.",
        "",
        f"Target: {_md(protocol['target']['name'])}; unit: {_md(protocol['target']['unit'])}; declared transformation: {_md(protocol['target']['transformation'])}.",
        "",
        f"Development ends {protocol['development_end']}; holdout begins {protocol['holdout_start']}. Horizon: {protocol['horizon']} month(s). Final training target cutoff: {protocol['final_training_cutoff']}.",
        "",
        "## Coverage",
        "",
        "| Scope | Raw months | Usable months | Excluded months |",
        "|---|---:|---:|---:|",
        f"| All input | {coverage['raw']} | {coverage['usable']} | {coverage['excluded']} |",
        *[
            f"| {name.title()} | {coverage[name]['raw']} | {coverage[name]['usable']} | {coverage[name]['excluded']} |"
            for name in ("development", "holdout")
        ],
        "",
        f"Development scoring uses {coverage['development']['validation_usable']} common usable months across {protocol['n_splits']} calendar folds. Every successful candidate and baseline has the same denominator. See input_rows.csv for each warmup or missing-feature exclusion.",
        "",
        "## Development candidates",
        "",
        "Candidates retain stable identifiers; table order is not a ranking. Selection uses pooled development MAE among successful OLS candidates only. Baselines are comparisons. A candidate failing any fold or the final fit cannot be selected.",
        "",
        "| Model | Role | Status | n | MAE | RMSE | Bias |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for candidate in result["candidates"]:
        metric = candidate["development"] or {}
        lines.append(
            f"| {candidate['id']} | {candidate['role']} | {candidate['status']} | {_display(metric.get('n'))} | {_display(metric.get('mae'))} | {_display(metric.get('rmse'))} | {_display(metric.get('bias'))} |"
        )
    lines += [
        "",
        "Positive bias means overprediction; negative bias means underprediction. Original-unit coefficients, training scales, R2, adjusted R2, condition number and VIF are in candidates.csv and result.json. Per-fold fits and exact calendar memberships are in folds.csv.",
        "",
    ]
    if result["stage"] == "holdout":
        lines += [
            "## Revealed holdout",
            "",
            "The model selected above is unchanged. Holdout results must not be used to choose a new winner. Coefficients remain fixed; the persistence baseline uses actual history as it becomes available at each later prediction origin.",
            "",
            "| Model | n | MAE | RMSE | Bias | Unavailable reason |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for candidate in result["candidates"]:
            metric = candidate["holdout"] or {}
            reason = candidate["holdout_reason"] or (
                candidate["reason"] if candidate["status"] == "failed" else ""
            )
            lines.append(
                f"| {candidate['id']} | {_display(metric.get('n'))} | {_display(metric.get('mae'))} | {_display(metric.get('rmse'))} | {_display(metric.get('bias'))} | {_md(reason)} |"
            )
        lines.append("")
    else:
        lines += [
            "## Holdout remains unscored",
            "",
            "No holdout predictions or metrics are exported. The complete raw observations remain in request.json and input_rows.csv for reproducibility. Reveal only after fixing the development decision, by rerunning with --reveal-holdout and a new output directory.",
            "",
        ]
    failed = [candidate for candidate in result["candidates"] if candidate["status"] == "failed"]
    lines += ["## Failed candidates and row exclusions", ""]
    if failed:
        lines += [f"- {candidate['id']}: {_md(candidate['reason'])}" for candidate in failed]
    else:
        lines += [
            "All enumerated candidates completed the required development folds and final training fit."
        ]
    lines += [
        "",
        "failures.csv preserves all model, fold, row-exclusion and revealed-holdout failure records. A populated final fit on a failed candidate does not cancel an earlier fold failure.",
        "",
        "## Reproduce and verify",
        "",
        f"Input source: {_md(source['name'])} ({source['kind']}); source SHA256: `{source['sha256']}`.",
        "",
        f"Selection fingerprint: `{result['selection_fingerprint']}`.",
        "",
        f"Complete data fingerprint: `{result['data_fingerprint']}`.",
        "",
        "Rerun this package's request.json in the same software environment with a new output directory. manifest.json binds the exact source request, normalized request, code, runtime and every other exported file. It does not contain absolute source paths. There is no network call or remote dependency in report viewing.",
        "",
        "## Method and limits",
        "",
    ]
    lines += [
        f"- {_md(protocol[key])}"
        for key in (
            "training_rule",
            "scaling_rule",
            "availability_rule",
            "sample_rule",
            "persistence_rule",
            "metrics_rule",
            "diagnostics_rule",
        )
    ]
    lines += [f"- {_md(value)}" for value in protocol["disclosures"]]
    if protocol["source_note"]:
        lines += ["", f"Caller source note: {_md(protocol['source_note'])}."]
    lines += [
        "",
        "Method references: [NumPy least squares](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html), [time-ordered splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html), [preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html). These references support the methods; they do not validate this dataset or application.",
        "",
    ]
    return "\n".join(lines).encode("utf-8")


def _bundle(normal: dict, result: dict, source: dict) -> dict[str, bytes]:
    candidates = []
    for candidate in result["candidates"]:
        row = {
            key: candidate[key]
            for key in ("id", "name", "role", "features", "status", "reason", "holdout_reason")
        }
        for split in ("development", "holdout"):
            for key in ("n", "mae", "rmse", "bias"):
                row[f"{split}_{key}"] = (candidate[split] or {}).get(key)
        fit = candidate["fit"] or {}
        row.update(
            {
                f"fit_{key}": fit.get(key)
                for key in (
                    "n",
                    "train_start",
                    "train_end",
                    "intercept",
                    "coefficients",
                    "feature_means",
                    "feature_scales",
                    "target_mean",
                    "r2",
                    "adjusted_r2",
                    "condition_number",
                    "vif",
                )
            }
        )
        candidates.append(row)
    candidate_headers = list(candidates[0])
    model_ids = [candidate["id"] for candidate in result["candidates"]]
    predictions = [
        {key: row[key] for key in ("period", "actual", "split", "fold")} | row["predictions"]
        for row in result["predictions"]
    ]
    folds = []
    for fold in result["folds"]:
        item = {key: value for key, value in fold.items() if key != "development"}
        item.update(
            {
                f"development_{key}": (fold["development"] or {}).get(key)
                for key in ("n", "mae", "rmse", "bias")
            }
        )
        folds.append(item)
    failures = []
    for candidate in result["candidates"]:
        if candidate["reason"]:
            failures.append(
                {"scope": "model", "model_id": candidate["id"], "reason": candidate["reason"]}
            )
        if candidate["holdout_reason"]:
            failures.append(
                {
                    "scope": "holdout",
                    "model_id": candidate["id"],
                    "reason": candidate["holdout_reason"],
                }
            )
    for fold in result["folds"]:
        if fold["reason"]:
            failures.append(
                {
                    "scope": "fold",
                    "model_id": fold["model_id"],
                    "fold": fold["fold"],
                    "reason": fold["reason"],
                }
            )
    for row in result["input_rows"]:
        for reason in row["reasons"]:
            failures.append({"scope": "input", "period": row["period"], **reason})
    files = {
        "request.json": _json(normal),
        "result.json": _json(result),
        "candidates.csv": _csv(candidate_headers, candidates),
        "folds.csv": _csv(list(folds[0]), folds),
        "predictions.csv": _csv(["period", "actual", "split", "fold", *model_ids], predictions),
        "input_rows.csv": _csv(list(result["input_rows"][0]), result["input_rows"]),
        "failures.csv": _csv(
            ["scope", "model_id", "fold", "period", "feature", "code", "source_period", "reason"],
            failures,
        ),
        "REPORT.md": _report(result, source),
    }
    manifest = {
        "schema_version": 1,
        "stage": result["stage"],
        "selection_fingerprint": result["selection_fingerprint"],
        "data_fingerprint": result["data_fingerprint"],
        "request_source": source,
        "software": {
            "package": "model-risk-lab",
            "version": __version__,
            "python": ".".join(map(str, sys.version_info[:3])),
            "numpy": np.__version__,
        },
        "code_sha256": {
            "forecast.py": _sha(Path(forecast.__file__).read_bytes()),
            "forecast_cli.py": _sha(Path(__file__).read_bytes()),
        },
        "files_sha256": {name: _sha(content) for name, content in sorted(files.items())},
        "notes": [
            "Hashes cover every package file except this manifest itself.",
            "request.json is sorted and normalized; request_source.sha256 binds the original source bytes before normalization.",
            "CSV text beginning with formula-significant characters has a protective apostrophe. JSON preserves its exact text.",
            "No timestamps or output paths are embedded, so the same input and runtime produce deterministic files.",
        ],
    }
    files["manifest.json"] = _json(manifest)
    return files


def _publish_exclusive(source: Path, destination: Path) -> None:
    """Atomically publish a same-filesystem directory without replacing any name."""
    if os.name == "nt":
        os.rename(source, destination)  # Windows rename refuses an existing destination.
        return
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(library, "renamex_np"):
        rename = library.renamex_np
        rename.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        rename.restype = ctypes.c_int
        status = rename(os.fsencode(source), os.fsencode(destination), 4)  # RENAME_EXCL
    elif sys.platform.startswith("linux") and hasattr(library, "renameat2"):
        rename = library.renameat2
        rename.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename.restype = ctypes.c_int
        status = rename(
            -100, os.fsencode(source), -100, os.fsencode(destination), 1
        )  # RENAME_NOREPLACE
    else:
        raise OSError(
            errno.ENOTSUP, "Atomic publication without replacement is unavailable on this platform"
        )
    if status != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def _write(files: dict[str, bytes], output: Path) -> Path:
    if os.path.lexists(output):
        raise ValueError("Output already exists; choose a new directory")
    parent = output.parent.resolve(strict=True)
    output = parent / output.name
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=parent))
    try:
        for name, content in files.items():
            with (staging / name).open("xb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
        _publish_exclusive(staging, output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError(f"Nonstandard JSON numeric constant: {value}")


def _read_request(path: Path) -> tuple[dict, dict]:
    before = path.stat()
    if not path.is_file() or before.st_size > MAX_REQUEST_BYTES:
        raise ValueError("Request must be a regular JSON file of at most 10 MiB")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        content = stream.read(MAX_REQUEST_BYTES + 1)
    after = path.stat()

    def identity(value: os.stat_result) -> tuple[int, int, int, int]:
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns

    if (
        len(content) > MAX_REQUEST_BYTES
        or identity(before) != identity(opened)
        or identity(before) != identity(after)
    ):
        raise ValueError("Request changed while being read or exceeded the 10 MiB limit")
    try:
        request = json.loads(
            content.decode("utf-8-sig"),
            object_pairs_hook=_unique_object,
            parse_constant=_invalid_constant,
        )
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError(
            f"Request is not a valid bounded UTF-8 JSON document: {type(error).__name__}"
        ) from None
    return request, {
        "kind": "file",
        "name": path.name,
        "sha256": _sha(content),
        "bytes": len(content),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Screen monthly OLS candidates and export an offline evidence package"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--example", action="store_true", help="Use 180 invented months and five features"
    )
    source.add_argument("--request", type=Path, help="Read an explicit monthly request JSON file")
    parser.add_argument(
        "--output", required=True, type=Path, help="A new output directory; its parent must exist"
    )
    parser.add_argument(
        "--reveal-holdout",
        action="store_true",
        help="Evaluate holdout after the development choice is fixed",
    )
    arguments = parser.parse_args(argv)
    try:
        output = arguments.output.absolute()
        if os.path.lexists(output):
            raise ValueError("Output already exists; choose a new directory")
        if arguments.example:
            request = forecast.example_request()
            content = _json(request)
            provenance = {
                "kind": "synthetic_example",
                "name": "example_request()",
                "sha256": _sha(content),
                "bytes": len(content),
            }
        else:
            request, provenance = _read_request(arguments.request)
        normal = forecast._normalise(request)
        result = forecast.run_experiment(normal, reveal_holdout=arguments.reveal_holdout)
        files = _bundle(normal, result, provenance)
        published = _write(files, output)
        selected = next(
            (
                candidate
                for candidate in result["candidates"]
                if candidate["id"] == result["selected_on_development"]
            ),
            None,
        )
        status = (
            1
            if selected is None or (arguments.reveal_holdout and selected["holdout"] is None)
            else 0
        )
        print(
            json.dumps(
                {
                    "output": str(published),
                    "stage": result["stage"],
                    "selected_on_development": result["selected_on_development"],
                    "selection_fingerprint": result["selection_fingerprint"],
                    "data_fingerprint": result["data_fingerprint"],
                    "exit_code": status,
                },
                sort_keys=True,
            )
        )
        return status
    except (OSError, ValueError) as error:
        print(f"Input/output error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
