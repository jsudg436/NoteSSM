from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def load_aggregate(path: Path, backbone: str, head: str, settings: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregate metrics file: {path}")
    frame = pd.read_csv(path)
    required = {"setting", "split", "AUPRC_mean", "AUROC_mean", "F1_mean", "Brier_mean", "ECE_mean"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    frame = frame[frame["setting"].isin(settings)].copy()
    frame["backbone"] = backbone
    frame["head"] = head
    frame["source_file"] = str(path)
    return frame


def metric(value: float, std: float | None = None) -> str:
    if std is None or pd.isna(std):
        return f"{value:.4f}"
    return f"{value:.4f} +/- {std:.4f}"


def write_markdown(summary: pd.DataFrame, output: Path) -> None:
    test = summary[summary["split"] == "test"].copy()
    test = test.sort_values(["backbone", "setting"])
    rows = [
        "# No-DAPT Baseline Comparison",
        "",
        "This comparison uses the frozen LAMP-R v0.2 benchmark and full 4096-token note-level cached embeddings.",
        "It is text-free and contains only aggregate metrics.",
        "",
        "## Test Metrics",
        "",
        "| Backbone | Head | Setting | AUPRC | AUROC | Precision | Recall | F1 | Brier | ECE | Threshold |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test.iterrows():
        rows.append(
            "| {backbone} | {head} | {setting} | {auprc} | {auroc} | {precision:.4f} | {recall:.4f} | {f1} | {brier:.4f} | {ece:.4f} | {threshold} |".format(
                backbone=row["backbone"],
                head=row["head"],
                setting=row["setting"].replace("_", " + "),
                auprc=metric(float(row["AUPRC_mean"]), float(row["AUPRC_std"]) if "AUPRC_std" in row else None),
                auroc=metric(float(row["AUROC_mean"]), float(row["AUROC_std"]) if "AUROC_std" in row else None),
                precision=float(row["Precision_mean"]),
                recall=float(row["Recall_mean"]),
                f1=metric(float(row["F1_mean"]), float(row["F1_std"]) if "F1_std" in row else None),
                brier=float(row["Brier_mean"]),
                ece=float(row["ECE_mean"]),
                threshold=metric(float(row["threshold_mean"]), float(row["threshold_std"]) if "threshold_std" in row else None),
            )
        )
    rows.extend(
        [
            "",
            "## Interpretation Rules",
            "",
            "- AUPRC and AUROC are computed from raw probabilities and are not threshold-tuned.",
            "- Precision, recall, and F1 use the validation-selected threshold.",
            "- Brier and ECE are computed from raw probabilities, not thresholded predictions.",
            "- Natural BCE should be treated as the calibrated-risk setting.",
            "- Natural class-weighted BCE should be treated as the discrimination/F1 setting when it wins on validation/test discrimination.",
        ]
    )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare frozen Mamba2 and ClinicalMamba no-DAPT baseline aggregates.")
    parser.add_argument("--mamba2-mlp-aggregate", required=True)
    parser.add_argument("--clinicalmamba-mlp-aggregate", required=True)
    parser.add_argument("--output-dir", default="reports/no_dapt_baseline_freeze")
    parser.add_argument("--settings", nargs="+", default=["natural_bce", "natural_class_weight"])
    args = parser.parse_args()

    settings = set(args.settings)
    frames = [
        load_aggregate(resolve_path(args.mamba2_mlp_aggregate), "mamba2_130m", "mlp", settings),
        load_aggregate(resolve_path(args.clinicalmamba_mlp_aggregate), "clinical_mamba_130m", "mlp", settings),
    ]
    summary = pd.concat(frames, ignore_index=True)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "no_dapt_baseline_comparison_metrics.csv"
    md_path = output_dir / "NO_DAPT_BASELINE_COMPARISON.md"
    summary.to_csv(csv_path, index=False)
    write_markdown(summary, md_path)
    print(summary.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
