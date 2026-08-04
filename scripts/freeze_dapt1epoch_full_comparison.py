from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HELPER_SPEC = importlib.util.spec_from_file_location(
    "freeze_no_dapt_full_comparison",
    ROOT / "scripts" / "freeze_no_dapt_full_comparison.py",
)
helper = importlib.util.module_from_spec(HELPER_SPEC)
assert HELPER_SPEC.loader is not None
sys.modules[HELPER_SPEC.name] = helper
HELPER_SPEC.loader.exec_module(helper)


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def fmt_mean_std(row: pd.Series, metric: str) -> str:
    return f"{float(row[f'{metric}_mean']):.4f} +/- {float(row[f'{metric}_std']):.4f}"


def bootstrap_row(bootstrap: pd.DataFrame, comparison: str, metric: str) -> pd.Series:
    row = bootstrap[(bootstrap["comparison"] == comparison) & (bootstrap["metric"] == metric)]
    if row.empty:
        raise ValueError(f"Missing bootstrap row for comparison={comparison}, metric={metric}")
    return row.iloc[0]


def ci_overlaps_zero(row: pd.Series) -> bool:
    return float(row["ci_low"]) <= 0.0 <= float(row["ci_high"])


def compact_delta_rows(bootstrap: pd.DataFrame) -> list[dict[str, object]]:
    metrics = ["AUPRC", "AUROC", "F1", "Recall", "Brier", "ECE"]
    rows = []
    for comparison, frame in bootstrap.groupby("comparison", sort=False):
        row: dict[str, object] = {"comparison": comparison}
        for metric in metrics:
            metric_row = frame[frame["metric"] == metric]
            row[metric] = float(metric_row["observed_delta_left_minus_right"].iloc[0]) if not metric_row.empty else np.nan
        rows.append(row)
    return rows


def dapt_metric_relation(main: pd.DataFrame, comparator_key: str, metrics: list[str]) -> str:
    test = main[main["split"] == "test"]
    dapt = test[test["model_key"] == "dapt1epoch_mamba2_4096"].iloc[0]
    comparator = test[test["model_key"] == comparator_key].iloc[0]
    deltas = {metric: float(dapt[f"{metric}_mean"] - comparator[f"{metric}_mean"]) for metric in metrics}
    if all(value < 0 for value in deltas.values()):
        return "below"
    if all(value > 0 for value in deltas.values()):
        return "above"
    return "mixed"


def interpretation_lines(main: pd.DataFrame, bootstrap: pd.DataFrame) -> list[str]:
    lines = []
    dapt_vs_mamba = "dapt1epoch_mamba2_4096_vs_mamba2_4096"
    ranking_metrics = ["AUPRC", "AUROC", "F1"]
    ranking_overlap = all(ci_overlaps_zero(bootstrap_row(bootstrap, dapt_vs_mamba, metric)) for metric in ranking_metrics)
    if ranking_overlap:
        lines.append("- DAPT does not show paired-bootstrap-supported ranking/discrimination improvement over no-DAPT Mamba2.")
    else:
        supported = [
            metric
            for metric in ranking_metrics
            if float(bootstrap_row(bootstrap, dapt_vs_mamba, metric)["ci_low"]) > 0
        ]
        if supported:
            lines.append(f"- DAPT shows paired-bootstrap-supported improvement over no-DAPT Mamba2 for: {', '.join(supported)}.")
        else:
            lines.append("- DAPT ranking/discrimination deltas versus no-DAPT Mamba2 are not uniformly favorable.")

    recall = bootstrap_row(bootstrap, dapt_vs_mamba, "Recall")
    if float(recall["observed_delta_left_minus_right"]) > 0 and float(recall["ci_low"]) > 0:
        lines.append("- DAPT significantly improves recall over no-DAPT Mamba2.")
    elif ci_overlaps_zero(recall):
        lines.append("- DAPT recall improvement over no-DAPT Mamba2 is not paired-bootstrap-supported.")
    else:
        lines.append("- DAPT does not improve recall over no-DAPT Mamba2.")

    clinical_relation = dapt_metric_relation(main, "clinicalmamba_4096", ranking_metrics)
    longformer_relation = dapt_metric_relation(main, "longformer_mean_4096", ranking_metrics)
    if clinical_relation == "below" and longformer_relation == "below":
        lines.append("- DAPT Mamba2 remains substantially below ClinicalMamba and Longformer in AUPRC/AUROC/F1.")
    else:
        lines.append(
            f"- DAPT Mamba2 relation on AUPRC/AUROC/F1: ClinicalMamba={clinical_relation}, Longformer={longformer_relation}."
        )

    bioclinical_relation = dapt_metric_relation(main, "bioclinicalbert_512", ranking_metrics)
    if bioclinical_relation == "below":
        lines.append("- DAPT Mamba2 also remains below the cheap 512-token BioClinicalBERT baseline on the main discrimination metrics.")
    elif bioclinical_relation == "above":
        lines.append("- DAPT Mamba2 is above BioClinicalBERT on AUPRC/AUROC/F1.")
    else:
        lines.append("- DAPT Mamba2 has mixed AUPRC/AUROC/F1 results versus BioClinicalBERT.")

    if clinical_relation == "below" and longformer_relation == "below":
        lines.append("- Therefore strict train-only DAPT alone is not sufficient to match strong public clinical or long-context encoder baselines.")
    return lines


def write_markdown(main: pd.DataFrame, bootstrap: pd.DataFrame, output: Path) -> None:
    test = main[main["split"] == "test"].copy()
    rows = [
        "# DAPT 1epoch Full Comparison Frozen",
        "",
        "- Benchmark: LAMP-R v0.2 frozen",
        "- Input: discharge summary",
        "- Embedding: frozen cached note embedding",
        "- Head: MLP",
        "- Loss: natural BCE",
        "- Seeds: 17, 23, 42",
        "- Threshold: selected on validation F1 and applied to test",
        "- AUPRC/AUROC/Brier/ECE: computed from raw probabilities",
        "- Precision/Recall/F1: computed from validation-F1-tuned threshold",
        "- Delta convention: left model minus right model. Positive AUPRC/AUROC/F1/Precision/Recall favors the left model; negative Brier/ECE favors the left model.",
        "",
        "## Test Main Table",
        "",
        "| Model | Role | Max length | Pooling | AUPRC | AUROC | F1 | Precision | Recall | Brier | ECE |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in test.sort_values("AUPRC_mean", ascending=False).iterrows():
        rows.append(
            f"| {row['display_name']} | {row['role']} | {int(row['max_length'])} | {row['pooling']} | "
            f"{fmt_mean_std(row, 'AUPRC')} | {fmt_mean_std(row, 'AUROC')} | {fmt_mean_std(row, 'F1')} | "
            f"{float(row['Precision_mean']):.4f} | {float(row['Recall_mean']):.4f} | "
            f"{float(row['Brier_mean']):.4f} | {float(row['ECE_mean']):.4f} |"
        )

    rows.extend(
        [
            "",
            "## Compact Test Delta Table",
            "",
            "| Comparison | AUPRC delta | AUROC delta | F1 delta | Recall delta | Brier delta | ECE delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in compact_delta_rows(bootstrap):
        rows.append(
            f"| {row['comparison']} | {float(row['AUPRC']):+.4f} | {float(row['AUROC']):+.4f} | "
            f"{float(row['F1']):+.4f} | {float(row['Recall']):+.4f} | "
            f"{float(row['Brier']):+.4f} | {float(row['ECE']):+.4f} |"
        )

    rows.extend(
        [
            "",
            "## DAPT Paired Bootstrap",
            "",
            "| Comparison | Metric | Delta left-right | 95% CI | Excludes 0 |",
            "|---|---|---:|---:|---|",
        ]
    )
    focus = bootstrap[bootstrap["metric"].isin(["AUPRC", "AUROC", "F1", "Precision", "Recall", "Brier", "ECE"])].copy()
    for _, row in focus.iterrows():
        rows.append(
            f"| {row['left_model']} vs {row['right_model']} | {row['metric']} | "
            f"{float(row['observed_delta_left_minus_right']):+.4f} | "
            f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}] | {row['ci_excludes_zero']} |"
        )

    rows.extend(
        [
            "",
            "## Interpretation",
            "",
            *interpretation_lines(main, bootstrap),
        ]
    )
    output.write_text("\n".join(rows) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze DAPT 1epoch comparison against all frozen full no-DAPT baselines.")
    parser.add_argument("--mamba2-dir", default="results/downstream_head_mlp_3seed/mamba2_last_full_len4096_sharded_mlp_natural_bce")
    parser.add_argument("--dapt-mamba2-dir", default="results/downstream_head_dapt1epoch_mamba2_3seed/dapt1epoch_mamba2_last_full_len4096_mlp_natural_bce")
    parser.add_argument("--clinicalmamba-dir", default="results/downstream_head_clinicalmamba_3seed/clinicalmamba_last_full_len4096_sharded_mlp_natural_bce")
    parser.add_argument("--longformer-dir", default="results/downstream_head_longformer_mean_3seed/longformer_mean_full_len4096_sharded_mlp_natural_bce")
    parser.add_argument("--bioclinicalbert-dir", default="results/downstream_head_bioclinicalbert_mean_3seed/bioclinicalbert_mean_full_len512_sharded_mlp_natural_bce")
    parser.add_argument("--output-dir", default="results/dapt1epoch_full_comparison_frozen")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260714)
    parser.add_argument("--expected-val-n", type=int, default=29298)
    parser.add_argument("--expected-test-n", type=int, default=29460)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    specs = [
        helper.ModelSpec("mamba2_4096", "Mamba2 no-DAPT", "General Mamba no-DAPT baseline", "discharge summary", 4096, "last_token", resolve_path(args.mamba2_dir)),
        helper.ModelSpec("dapt1epoch_mamba2_4096", "Mamba2 1epoch DAPT", "Strict train-only DAPT Mamba2 baseline", "discharge summary", 4096, "last_token", resolve_path(args.dapt_mamba2_dir)),
        helper.ModelSpec("clinicalmamba_4096", "ClinicalMamba no-DAPT", "Main Mamba-family no-DAPT baseline", "discharge summary", 4096, "last_token", resolve_path(args.clinicalmamba_dir)),
        helper.ModelSpec("longformer_mean_4096", "Longformer no-DAPT", "Non-Mamba long-context comparator", "discharge summary", 4096, "mean", resolve_path(args.longformer_dir)),
        helper.ModelSpec("bioclinicalbert_512", "BioClinicalBERT no-DAPT", "Cheap clinical encoder baseline", "discharge summary", 512, "mean", resolve_path(args.bioclinicalbert_dir)),
    ]
    expected_eval_counts = {"val": args.expected_val_n, "test": args.expected_test_n}
    predictions = {}
    for spec in specs:
        for seed in args.seeds:
            predictions[(spec.key, seed)] = helper.load_predictions(spec, seed, expected_eval_counts)

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    main_table = helper.main_table(specs, args.seeds, predictions)
    main_table.to_csv(output_dir / "dapt1epoch_full_main_table.csv", index=False)

    calibration = helper.calibration_summary(specs, args.seeds, predictions)
    calibration.to_csv(output_dir / "dapt1epoch_full_calibration_summary.csv", index=False)

    errors = helper.error_summary(specs, args.seeds, predictions)
    errors.to_csv(output_dir / "dapt1epoch_full_error_summary.csv", index=False)

    rng = np.random.default_rng(args.bootstrap_seed)
    comparisons = [
        ("dapt1epoch_mamba2_4096", "mamba2_4096"),
        ("dapt1epoch_mamba2_4096", "clinicalmamba_4096"),
        ("dapt1epoch_mamba2_4096", "longformer_mean_4096"),
        ("dapt1epoch_mamba2_4096", "bioclinicalbert_512"),
    ]
    boot_rows = []
    for comparison in comparisons:
        boot_rows.extend(helper.paired_bootstrap(comparison, args.seeds, predictions, args.n_bootstrap, rng))
    bootstrap = pd.DataFrame(boot_rows)
    bootstrap.to_csv(output_dir / "dapt1epoch_full_delta_bootstrap.csv", index=False)

    write_markdown(main_table, bootstrap, output_dir / "DAPT1EPOCH_FULL_COMPARISON_FROZEN.md")

    print(f"Wrote {output_dir / 'DAPT1EPOCH_FULL_COMPARISON_FROZEN.md'}")
    print(f"Wrote {output_dir / 'dapt1epoch_full_main_table.csv'}")
    print(f"Wrote {output_dir / 'dapt1epoch_full_delta_bootstrap.csv'}")
    print(f"Wrote {output_dir / 'dapt1epoch_full_calibration_summary.csv'}")
    print(f"Wrote {output_dir / 'dapt1epoch_full_error_summary.csv'}")
    print(main_table[main_table["split"] == "test"].sort_values("AUPRC_mean", ascending=False).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
