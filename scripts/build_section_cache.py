from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.section_cache import SectionLengthConfig, build_section_cache_for_split  # noqa: E402


def load_tokenizer(tokenizer_id: str | None):
    if not tokenizer_id:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_id)


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build restricted local section cache artifacts from frozen LAMP-R parquets.")
    parser.add_argument("--benchmark-root", default="results/lampr_full_v0_2")
    parser.add_argument("--output-dir", default="results/lampr_full_v0_2/section_cache")
    parser.add_argument("--tokenizer-id", default=None)
    parser.add_argument("--section-max-tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-split row limit for smoke tests.")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer_id)
    config = SectionLengthConfig(section_max_tokens=args.section_max_tokens)
    benchmark_root = resolve_path(args.benchmark_root)
    output_dir = resolve_path(args.output_dir)
    for split in ["train", "val", "test"]:
        path = build_section_cache_for_split(
            benchmark_root=benchmark_root,
            split=split,
            output_dir=output_dir,
            tokenizer=tokenizer,
            limit=args.limit,
            length_config=config,
        )
        print(f"Wrote {path}")
        print(f"Wrote {path.parent / f'fallback_chunk_cache_{split}.parquet'}")
    print("Section and fallback chunk cache artifacts are restricted local files because they contain text.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
