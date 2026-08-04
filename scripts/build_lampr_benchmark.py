from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.lampr_builder import build_lampr, read_table  # noqa: E402
from notesm_v2.utils.config import load_yaml, require_config_paths  # noqa: E402


def resolve_output_path(root: Path, configured: str) -> Path:
    path = Path(configured)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == root.name:
        return root.parent / path
    if Path.cwd().resolve() == root.resolve():
        return Path.cwd() / path
    return root / path


def write_split_outputs(output_dir: Path, split_examples: pd.DataFrame) -> None:
    for split in ["train", "val", "test"]:
        subset = split_examples[split_examples["split"] == split].copy() if "split" in split_examples else split_examples.iloc[0:0].copy()
        subset.to_parquet(output_dir / f"benchmark_examples_{split}.parquet", index=False)
        subjects = subset[["subject_id"]].drop_duplicates().sort_values("subject_id")
        subjects.to_csv(output_dir / f"split_subjects_{split}.csv", index=False)


def write_summary(output_dir: Path, result) -> None:
    final_count = len(result.split_examples)
    positive = int((result.split_examples["label"] == 1).sum()) if final_count else 0
    negative = int((result.split_examples["label"] == 0).sum()) if final_count else 0
    dapt_blocks = len(result.dapt_train_manifest)
    dapt_patients = int(result.dapt_train_sequences["subject_id_hash"].nunique()) if not result.dapt_train_sequences.empty else 0
    lines = [
        "# LAMP-R Benchmark Run Summary",
        "",
        "This summary intentionally excludes raw clinical note text.",
        "",
        "## Final Labels",
        "",
        f"- Final labeled examples: {final_count}",
        f"- Positive examples: {positive}",
        f"- Negative examples: {negative}",
        "",
        "## DAPT Artifacts",
        "",
        f"- Train-only DAPT patients: {dapt_patients}",
        f"- Train-only DAPT manifest blocks: {dapt_blocks}",
        "- `dapt_train_sequences.parquet` is a restricted local artifact because it contains note text sequences.",
        "",
        "## Output Files",
        "",
        "- `benchmark_flow_counts.csv`",
        "- `benchmark_flow_counts_priority.csv`",
        "- `benchmark_filter_counts.csv`",
        "- `label_counts.csv`",
        "- `split_counts.csv`",
        "- `note_time_filter_counts.csv`",
        "- `death_censoring_counts.csv`",
        "- `planned_admission_counts.csv`",
        "- `section_parser_stats.csv`",
        "- `benchmark_manifest_no_text.parquet`",
        "- `dapt_train_manifest.parquet`",
        "- `dapt_train_sequences.parquet`",
    ]
    (output_dir / "benchmark_run_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the LAMP-R leakage-aware benchmark.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "data_lampr.yaml"))
    parser.add_argument("--debug-sample", type=int, default=None, help="Limit rows from each raw input for schema/debug checks.")
    parser.add_argument("--output-dir", default=None, help="Override config output.benchmark_dir.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    require_config_paths(config, ["paths.admissions", "paths.patients", "paths.discharge_notes"])

    admissions = read_table(config["paths"]["admissions"])
    patients = read_table(config["paths"]["patients"])
    notes = read_table(config["paths"]["discharge_notes"])
    if args.debug_sample:
        admissions = admissions.head(args.debug_sample).copy()
        patients = patients[patients["subject_id"].isin(admissions["subject_id"])].copy() if "subject_id" in patients else patients.head(args.debug_sample).copy()
        notes = notes[notes["subject_id"].isin(admissions["subject_id"])].head(args.debug_sample).copy() if "subject_id" in notes else notes.head(args.debug_sample).copy()

    configured_output = args.output_dir or config["output"]["benchmark_dir"]
    output_dir = resolve_output_path(ROOT, configured_output)
    reports_dir = resolve_output_path(ROOT, config["output"]["reports_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    if "admission_type" in admissions.columns:
        counts = admissions["admission_type"].fillna("MISSING").value_counts().rename_axis("admission_type")
        counts.to_frame("count").to_csv(reports_dir / "admission_type_counts.csv")

    result = build_lampr(admissions=admissions, patients=patients, notes=notes, config=config)

    result.filter_counts.to_csv(output_dir / "benchmark_filter_counts.csv", index=False)
    result.label_counts.to_csv(output_dir / "label_counts.csv", index=False)
    result.split_counts.to_csv(output_dir / "split_counts.csv", index=False)
    result.episode_merge_counts.to_csv(output_dir / "episode_merge_counts.csv", index=False)
    result.timing_counts.to_csv(output_dir / "note_timing_and_censor_counts.csv", index=False)
    result.benchmark_flow_counts.to_csv(output_dir / "benchmark_flow_counts.csv", index=False)
    result.benchmark_flow_counts_priority.to_csv(output_dir / "benchmark_flow_counts_priority.csv", index=False)
    result.note_time_filter_counts.to_csv(output_dir / "note_time_filter_counts.csv", index=False)
    result.death_censoring_counts.to_csv(output_dir / "death_censoring_counts.csv", index=False)
    result.planned_admission_counts.to_csv(output_dir / "planned_admission_counts.csv", index=False)
    result.section_parser_stats.to_csv(output_dir / "section_parser_stats.csv", index=False)
    result.benchmark_manifest_no_text.to_parquet(output_dir / "benchmark_manifest_no_text.parquet", index=False)
    result.dapt_train_manifest.to_parquet(output_dir / "dapt_train_manifest.parquet", index=False)
    result.dapt_train_sequences.to_parquet(output_dir / "dapt_train_sequences.parquet", index=False)
    write_split_outputs(output_dir, result.split_examples)
    write_summary(output_dir, result)

    print(f"Wrote benchmark outputs to {output_dir}")
    print(f"Final labeled examples: {len(result.split_examples)}")
    if not result.split_counts.empty:
        print(result.split_counts.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
