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


def parse_setting(path: Path, prefix: str) -> str:
    name = path.name
    return name[len(prefix) :] if prefix and name.startswith(prefix) else name


def read_seed_metrics(run_dir: Path, model: str, seed: int, tuned: bool) -> pd.DataFrame:
    if tuned:
        path = run_dir / f"{model}_threshold_tuned_metrics_seed{seed}.csv"
    else:
        path = run_dir / f"{model}_cached_embedding_metrics_seed{seed}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    frame = pd.read_csv(path)
    frame["seed"] = seed
    frame["metrics_file"] = str(path)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate cached-head 3-seed downstream metrics.")
    parser.add_argument("--root", required=True, help="Directory containing one subdirectory per setting.")
    parser.add_argument("--setting-prefix", default="")
    parser.add_argument("--model", default="notesm_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[17, 23, 42])
    parser.add_argument("--tuned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-prefix", default=None)
    args = parser.parse_args()

    root = resolve_path(args.root)
    run_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    if args.setting_prefix:
        run_dirs = [path for path in run_dirs if path.name.startswith(args.setting_prefix)]
    if not run_dirs:
        raise ValueError(f"No run directories found under {root}")

    frames = []
    for run_dir in run_dirs:
        setting = parse_setting(run_dir, args.setting_prefix)
        for seed in args.seeds:
            frame = read_seed_metrics(run_dir, args.model, seed, args.tuned)
            frame["setting"] = setting
            frame["run_dir"] = str(run_dir)
            frames.append(frame)
    per_seed = pd.concat(frames, ignore_index=True)
    metric_cols = [
        col
        for col in ["AUPRC", "AUROC", "Accuracy", "Precision", "Recall", "F1", "Brier", "ECE", "threshold"]
        if col in per_seed.columns
    ]
    aggregate = (
        per_seed.groupby(["setting", "split"], as_index=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate.columns = ["_".join(col).rstrip("_") if isinstance(col, tuple) else col for col in aggregate.columns]
    prefix = args.output_prefix or ("threshold_tuned" if args.tuned else "threshold_0p5")
    per_seed_path = root / f"{prefix}_per_seed_metrics.csv"
    aggregate_path = root / f"{prefix}_aggregate_metrics.csv"
    per_seed.to_csv(per_seed_path, index=False)
    aggregate.to_csv(aggregate_path, index=False)
    print("Per-seed metrics:")
    print(per_seed[["setting", "seed", "split", *metric_cols]].to_string(index=False))
    print("\nAggregate metrics:")
    print(aggregate.to_string(index=False))
    print(f"Wrote {per_seed_path}")
    print(f"Wrote {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
