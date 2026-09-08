"""Generate a portable validation report from fresh synthetic inputs, entirely offline."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import __version__
from .credit import BASELINES, CANDIDATES, evaluate, synthetic_quarters
from .fx import convergence_experiment

matplotlib.rcParams.update({"svg.hashsalt": "model-risk-lab", "svg.fonttype": "none",
                           "font.family": "DejaVu Sans", "font.size": 10,
                           "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path: Path) -> None:
    fig.savefig(path, bbox_inches="tight", metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


EVIDENCE_FILES = (
    "REPORT.md", "credit_predictions.csv", "credit_protocol.svg", "credit_validation.svg",
    "fx_bumps.csv", "fx_convergence.svg", "results.json", "synthetic_quarters.csv",
)


def verify_bundle(output: Path) -> dict[str, str]:
    """Check all bundle bytes; hashes detect changes, not authorship or correctness."""
    output = Path(output)
    expected = set(EVIDENCE_FILES) | {"checksums.json"}
    if output.is_symlink() or not output.is_dir():
        raise ValueError("report must be a regular directory")
    if {p.name for p in output.iterdir()} != expected:
        raise ValueError("report file inventory is incomplete or contains unexpected files")
    for name in expected:
        if (output / name).is_symlink() or not (output / name).is_file():
            raise ValueError(f"report evidence must be a regular file: {name}")
    try:
        manifest = json.loads((output / "checksums.json").read_text())
        if not isinstance(manifest, dict):
            raise ValueError("checksum manifest must be an object")
        hashes = manifest["sha256"]
        if (type(manifest["schema_version"]) is not int or manifest["schema_version"] != 1
                or not isinstance(hashes, dict) or set(hashes) != set(EVIDENCE_FILES)):
            raise ValueError("unsupported checksum manifest")
        if any(not isinstance(digest, str) or len(digest) != 64
               or any(ch not in "0123456789abcdef" for ch in digest) for digest in hashes.values()):
            raise ValueError("checksum manifest requires lowercase SHA-256 strings")
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid checksum manifest") from exc
    for name, digest in hashes.items():
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"checksum mismatch: {name}")
    return hashes


def generate(output: Path, seed: int = 20260907) -> dict:
    """Build in a sibling staging directory and publish only a complete new bundle."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists; choose a new directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        stage = Path(temporary) / "report"
        stage.mkdir()
        summary = _generate(stage, seed)
        hashes = {name: hashlib.sha256((stage / name).read_bytes()).hexdigest()
                  for name in EVIDENCE_FILES}
        (stage / "checksums.json").write_text(
            json.dumps({"schema_version": 1, "sha256": hashes}, indent=2) + "\n")
        verify_bundle(stage)
        # Recheck after generation so an output created in the meantime is preserved.
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"output appeared during generation: {output}")
        stage.rename(output)
    return {**summary, "output": str(output)}


def _generate(output: Path, seed: int) -> dict:
    raw = synthetic_quarters(seed)
    raw.to_csv(output / "synthetic_quarters.csv", float_format="%.12g")
    credit = evaluate(seed)
    fx = convergence_experiment()
    predictions = pd.DataFrame(credit["predictions"]).set_index("quarter")
    predictions.to_csv(output / "credit_predictions.csv", float_format="%.12g")
    pd.DataFrame([{k: v for k, v in row.items() if k != "inputs"} for row in fx]).to_csv(
        output / "fx_bumps.csv", index=False, float_format="%.12g")
    versions = {p: importlib.metadata.version(p) for p in
                ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")}
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted(Path(__file__).parent.glob("*.py"))}
    manifest = {"package_version": __version__, "seed": seed,
                "input_kind": "fresh synthetic data, no calibration to external records",
                "dependencies": versions, "source_sha256": source_hashes,
                "input_sha256": hashlib.sha256((output / "synthetic_quarters.csv").read_bytes()).hexdigest()}
    result = {"manifest": manifest, "credit": credit, "fx": fx}
    (output / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    selected = credit["selected_on_development"]
    ax = axes[0]
    for name, label, color in [("actual", "Observed synthetic rate", "#172a3a"),
                                (selected, f"Selected: {selected}", "#007f73"),
                                ("last_quarter", "Last-quarter baseline", "#d57521")]:
        ax.plot(predictions.index, predictions[name]*100, label=label, color=color,
                linewidth=1.8, linestyle="--" if name == "last_quarter" else "-")
    ax.axvline(124, color="#8a4764", linestyle=":", label="Predefined regime change")
    ax.set(xlabel="Synthetic quarter", ylabel="Default rate (%)", title="Locked holdout: one-step forecasts")
    ax.legend(fontsize=8, loc="upper left")
    names = list(BASELINES + CANDIDATES)
    pos = np.arange(len(names))
    axes[1].barh(pos - 0.18, [credit["cv"][n]["mae_bp"] for n in names], height=.35,
                 label="Development fold-out MAE", color="#007f73")
    axes[1].barh(pos + 0.18, [credit["holdout"][n]["mae_bp"] for n in names], height=.35,
                 label="Holdout MAE (diagnostic)", color="#9ccbc3")
    axes[1].set(yticks=pos, yticklabels=names, xlabel="MAE (basis points)",
                title="Accuracy includes both simple baselines")
    axes[1].legend(fontsize=8)
    _save(fig, output / "credit_validation.svg")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), constrained_layout=True)
    folds = [r for r in credit["folds"] if r["model"] == "ols"]
    for i, row in enumerate(folds):
        axes[0].broken_barh([(row["train_start"], row["n_train"])], (i-.3, .6), facecolors="#9ccbc3")
        axes[0].broken_barh([(row["test_start"], row["n_test"])], (i-.3, .6), facecolors="#007f73")
    axes[0].axvspan(112, 148, color="#d57521", alpha=.15, label="Holdout (never used to tune)")
    axes[0].set(xlabel="Synthetic quarter", yticks=range(5), yticklabels=[f"Fold {i}" for i in range(1,6)],
                title="Rolling training windows / forward test blocks", xlim=(0, 148))
    axes[0].legend(fontsize=8, loc="upper left")
    for name, color in [("ols", "#007f73"), ("complex_ols", "#d57521")]:
        records = [r for r in credit["folds"] if r["model"] == name]
        axes[1].plot(range(1, 6), [r["train_r2_logit"]["value"] for r in records],
                     "o-", label=name, color=color)
    axes[1].set(xlabel="Development fold", ylabel="Training R² (logit scale)",
                xticks=range(1, 6), title="Additional noise features improve training fit")
    axes[1].legend(fontsize=8)
    _save(fig, output / "credit_protocol.svg")

    frame = pd.DataFrame(fx)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    subset = frame[(frame.case == "near_money") & (frame.kind == "call")]
    for greek in ("delta", "gamma", "vega", "rho_domestic", "rho_foreign"):
        r = subset[subset.greek == greek]
        axes[0].loglog(r.relative_bump, np.maximum(r.absolute_error, 1e-18), "o-", label=greek, markersize=3)
    axes[0].set(xlabel="Bump / predefined input scale", ylabel="Absolute derivative error (unit notional)",
                title="All bumps retained, including tiny-step cancellation")
    axes[0].legend(fontsize=7)
    gamma = subset[subset.greek == "gamma"]
    axes[1].semilogx(gamma.relative_bump, gamma.estimate, "o-", color="#007f73", label="Price-based central difference")
    axes[1].axhline(gamma.analytic.iloc[0], color="#d57521", linestyle="--", label="Analytical gamma")
    axes[1].set(xlabel="Relative spot bump", ylabel="Gamma (DOM per quote unit²)",
                title="Too small a bump can destroy the estimate")
    axes[1].legend(fontsize=8)
    _save(fig, output / "fx_convergence.svg")

    selected_mae = credit["holdout"][selected]["mae_bp"]
    baseline_mae = credit["holdout"]["last_quarter"]["mae_bp"]
    comparison = "lower" if selected_mae < baseline_mae else "higher"
    rows = ["| Model | Development MAE (bp) | Holdout MAE (bp) | Holdout bias (bp) |",
            "|:--|--:|--:|--:|"]
    for name in names:
        rows.append(f"| {name}{' **selected**' if name == selected else ''} | {credit['cv'][name]['mae_bp']:.2f} | "
                    f"{credit['holdout'][name]['mae_bp']:.2f} | {credit['holdout'][name]['bias_bp']:.2f} |")
    representative = frame[frame.relative_bump == 1e-4]
    fx_error_rows = ["| Greek | Max absolute error at relative bump 1e-4 |",
                     "|:--|--:|"]
    for greek, group in representative.groupby("greek", sort=False):
        fx_error_rows.append(f"| {greek} | {group.absolute_error.max():.3g} |")
    (output / "REPORT.md").write_text(f"""# Validation report — seed {seed}

Generated by Model Risk Lab {__version__}. All observations and option scenarios are invented.
Reproduce with `model-risk-lab --seed {seed} --output outputs/reproduction`.
Credit inputs are in [synthetic_quarters.csv](synthetic_quarters.csv). The generating mechanism is
documented in the repository's credit method note. Fold metrics, FX scenario parameters, versions
and code hashes are in [results.json](results.json). The eight evidence files are covered by
[checksums.json](checksums.json); check this bundle with `model-risk-lab --verify PATH_TO_REPORT`.
Hashes detect changes to a bundle; they do not certify model correctness or authorship.

## Credit regression

**{selected}** was selected using development fold-out MAE only. Its holdout MAE is
**{selected_mae:.2f} bp**, {comparison} than the last-quarter baseline's **{baseline_mae:.2f} bp**.
This is one synthetic experiment, not evidence of predictive skill on a real portfolio.

{chr(10).join(rows)}

Holdout scores for unselected candidates are diagnostic; they do not change the selected model.
Quarterly information is updated through each test block; coefficients stay fixed in the block.
This is not a multi-quarter forecast made at a single origin. The holdout includes a predefined
mechanism change at quarter 124. Bias means prediction minus observation.

![Holdout and errors](credit_validation.svg)
![Time protocol and training fit](credit_protocol.svg)

The deliberately invalid current-target feature obtained mean fold MAE
**{credit['invalid_protocol']['mean_fold_mae_bp']:.4f} bp**. Its information-availability check rejected it:
**{credit['invalid_protocol']['availability_check_rejected']}**. It is excluded from selection. A good-looking
score cannot repair invalid information timing. The small remaining error reflects the smoothed
target transformation, not a deployable prediction.

## FX sensitivities

The experiment retains **{len(fx)}** estimates: three synthetic scenarios, call and put,
five Greeks, and nine predetermined bump sizes. Full price-up/base/down values and errors
are in [fx_bumps.csv](fx_bumps.csv); scenario inputs are in [results.json](results.json).

At the predeclared relative bump 1e-4, maximum absolute errors over the six case/payoff combinations
are shown separately for each raw derivative, at unit foreign notional. Their units differ;
use the method note's unit table when interpreting them. No bump is selected after inspecting results.

{chr(10).join(fx_error_rows)}

![Difference convergence](fx_convergence.svg)

Analytical agreement checks implementation consistency under a flat-volatility model. It does
not validate market calibration or economic suitability. Unit contracts, mathematical identities,
and finite differences address different failure modes.

## Evidence and next experiment

- [Synthetic input CSV](synthetic_quarters.csv), [holdout predictions](credit_predictions.csv),
  [full result and provenance](results.json), [all FX differences](fx_bumps.csv).
- Use a new seed to test whether conclusions survive another invented path; do not select seeds for a flattering result.
- Next credit extension: publish-delay assumptions with revised vs first-release data semantics.
- Next FX extension: independently test tail stability and numerical pricing with convergence bounds.
- Educational scope: no real customer data, regulatory approval, production use or full ECL/SIMM implementation.
""")
    return {"output": str(output), "selected": selected, "holdout_mae_bp": selected_mae,
            "baseline_mae_bp": baseline_mae, "fx_estimates": len(fx)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260907)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, default=Path("outputs/latest"))
    mode.add_argument("--verify", type=Path, help="check a generated report's inventory and hashes")
    args = parser.parse_args()
    try:
        result = ({"verified": str(args.verify), "evidence_files": len(verify_bundle(args.verify))}
                  if args.verify is not None else generate(args.output, args.seed))
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
