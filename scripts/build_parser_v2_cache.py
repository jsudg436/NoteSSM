from __future__ import annotations

import argparse
import gc
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from notesm_v2.data.local_input_v2 import (  # noqa: E402
    LOCKED_AUGMENTED_TOKENIZER_JSON_SHA256,
    LOCKED_BASE_TOKENIZER_JSON_SHA256,
    sha256_path,
)
from notesm_v2.data.section_cache_v2 import (  # noqa: E402
    IMPLEMENTATION_STAGE_SPLIT,
    bind_locked_tokenizer_to_model,
    build_parser_v2_cache_frames,
    load_locked_augmented_tokenizer,
    validate_locked_bytelevel_offset_contract,
    verify_protocol_sidecar,
    write_parser_v2_cache,
)
from notesm_v2.data.section_parser_v2 import ParserV2Config  # noqa: E402


DEFAULT_PROTOCOL = "reports/PARSER_V2_K2_PROTOCOL_LOCKED_2026-07-30.md"
DEFAULT_OFFSET_AMENDMENT = (
    "reports/"
    "PARSER_V2_BYTELEVEL_ZERO_WIDTH_OFFSET_AMENDMENT_LOCKED_2026-07-31.md"
)
DEFAULT_BENCHMARK_ROOT = "results/lampr_full_v0_2"
DEFAULT_OUTPUT_DIR = (
    "results/parser_v2_implementation/"
    "parser_v2_train_cache_native_zero_width_r3_20260731"
)
LOCKED_MODEL_CONFIG_SHA256 = (
    "78bad17c8025c87f160927c811ba2af60c8e2bf279ea98394d5849f8fb4811e4"
)
# Reused from the SHA-gated authoritative K1-L v1.1/v1.2 runners for the
# identical checkpoint-step-14763 pytorch_training_state.pt artifact.
LOCKED_CHECKPOINT_STATE_SHA256 = (
    "d2023c6e77e871333e8ed461e8d804ed42e7fb3830149508b112526e335d7679"
)


def resolve_path(path_arg: str) -> Path:
    path = Path(path_arg)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == ROOT.name:
        return ROOT.parent / path
    if Path.cwd().resolve() == ROOT.resolve():
        return Path.cwd() / path
    return ROOT / path


def _validate_sha256_literal(value: str, name: str) -> str:
    normalized = value.strip().lower()
    if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return normalized


def _locked_code_hashes() -> dict[str, str]:
    paths = {
        "src/notesm_v2/data/section_parser_v2.py": (
            ROOT / "src" / "notesm_v2" / "data" / "section_parser_v2.py"
        ),
        "src/notesm_v2/data/local_input_v2.py": (
            ROOT / "src" / "notesm_v2" / "data" / "local_input_v2.py"
        ),
        "src/notesm_v2/data/section_cache_v2.py": (
            ROOT / "src" / "notesm_v2" / "data" / "section_cache_v2.py"
        ),
        "src/notesm_v2/utils/parser_v2_privacy.py": (
            ROOT / "src" / "notesm_v2" / "utils" / "parser_v2_privacy.py"
        ),
        "scripts/build_parser_v2_cache.py": Path(__file__).resolve(),
        "scripts/preflight_parser_v2_tokenizer.py": (
            ROOT / "scripts" / "preflight_parser_v2_tokenizer.py"
        ),
        "tests/test_section_parser_v2.py": (
            ROOT / "tests" / "test_section_parser_v2.py"
        ),
        "tests/test_local_input_v2.py": (
            ROOT / "tests" / "test_local_input_v2.py"
        ),
        "tests/test_section_cache_v2.py": (
            ROOT / "tests" / "test_section_cache_v2.py"
        ),
        "tests/test_parser_v2_privacy.py": (
            ROOT / "tests" / "test_parser_v2_privacy.py"
        ),
        "tests/test_parser_v2_zero_width_contract.py": (
            ROOT / "tests" / "test_parser_v2_zero_width_contract.py"
        ),
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("parser-v2 implementation code file is missing")
    return {name: sha256_path(path) for name, path in paths.items()}


def _read_train_examples(benchmark_root: Path) -> pd.DataFrame:
    train_path = benchmark_root / "benchmark_examples_train.parquet"
    if not train_path.is_file():
        raise FileNotFoundError("locked train benchmark parquet is missing")
    # This implementation stage intentionally does not request label, patient,
    # validation, or test columns/files.
    examples = pd.read_parquet(
        train_path,
        columns=["example_id", "text"],
    )
    examples = examples.copy()
    examples["split"] = IMPLEMENTATION_STAGE_SPLIT
    return examples[["example_id", "split", "text"]]


def _bind_tokenizer_to_locked_checkpoint(
    tokenizer_binding,
    *,
    model_dir: Path,
    checkpoint_dir: Path,
    expected_model_config_sha256: str,
    expected_checkpoint_state_sha256: str,
):
    model_config_path = model_dir / "config.json"
    checkpoint_state_path = checkpoint_dir / "pytorch_training_state.pt"
    if not model_config_path.is_file():
        raise FileNotFoundError("locked Mamba2 config.json is missing")
    if not checkpoint_state_path.is_file():
        raise FileNotFoundError("locked DAPT checkpoint state is missing")
    observed_model_config_sha = sha256_path(model_config_path)
    if observed_model_config_sha != expected_model_config_sha256:
        raise ValueError("locked Mamba2 config.json SHA-256 mismatch")
    observed_checkpoint_sha = sha256_path(checkpoint_state_path)
    if observed_checkpoint_sha != expected_checkpoint_state_sha256:
        raise ValueError("locked DAPT checkpoint SHA-256 mismatch")

    from transformers import AutoModelForCausalLM
    import torch

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        output_hidden_states=True,
    )
    if not hasattr(model, "get_input_embeddings") or not hasattr(
        model,
        "resize_token_embeddings",
    ):
        raise ValueError("locked Mamba2 model lacks embedding resize support")
    current_rows = int(model.get_input_embeddings().weight.shape[0])
    if current_rows != len(tokenizer_binding.tokenizer):
        model.resize_token_embeddings(len(tokenizer_binding.tokenizer))

    try:
        try:
            state = torch.load(
                checkpoint_state_path,
                map_location="cpu",
                weights_only=False,
                mmap=True,
            )
        except TypeError:
            state = torch.load(checkpoint_state_path, map_location="cpu")
    except Exception:
        raise ValueError("locked DAPT checkpoint could not be loaded") from None
    if not isinstance(state, dict) or not isinstance(state.get("model_state"), dict):
        raise ValueError("locked DAPT checkpoint lacks model_state")
    try:
        model.load_state_dict(state["model_state"], strict=True)
    except Exception:
        raise ValueError(
            "locked DAPT checkpoint is incompatible with the pinned model"
        ) from None

    model.eval()
    model_bound_binding = bind_locked_tokenizer_to_model(
        tokenizer_binding,
        model,
    )
    checkpoint_identity = {
        "model_config_sha256": observed_model_config_sha,
        "checkpoint_state_sha256": observed_checkpoint_sha,
        "checkpoint_global_step": state.get("global_step"),
        "checkpoint_format": state.get("checkpoint_format"),
        "model_embedding_rows": int(
            model.get_input_embeddings().weight.shape[0]
        ),
        "tokenizer_vocab_size": int(len(tokenizer_binding.tokenizer)),
        "strict_model_state_load": True,
    }
    del state
    del model
    gc.collect()
    return model_bound_binding, checkpoint_identity


def run_builder(args: argparse.Namespace) -> dict[str, object]:
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError("parser-v2 cache output directory already exists")

    protocol_path = resolve_path(args.protocol_path)
    protocol_sidecar = (
        resolve_path(args.protocol_sha256_path)
        if args.protocol_sha256_path
        else protocol_path.with_suffix(".sha256")
    )
    protocol_sha256 = verify_protocol_sidecar(protocol_path, protocol_sidecar)
    amendment_path = resolve_path(args.offset_amendment_path)
    amendment_sidecar = (
        resolve_path(args.offset_amendment_sha256_path)
        if args.offset_amendment_sha256_path
        else amendment_path.with_suffix(".sha256")
    )
    offset_amendment_sha256 = verify_protocol_sidecar(
        amendment_path,
        amendment_sidecar,
    )

    expected_base_sha = _validate_sha256_literal(
        args.expected_base_tokenizer_json_sha256,
        "expected base tokenizer.json SHA-256",
    )
    expected_augmented_sha = _validate_sha256_literal(
        args.expected_augmented_tokenizer_json_sha256,
        "expected augmented tokenizer.json SHA-256",
    )
    base_tokenizer_json = resolve_path(args.base_tokenizer_dir) / "tokenizer.json"
    if not base_tokenizer_json.is_file():
        raise FileNotFoundError("locked base tokenizer.json is missing")
    observed_base_sha = sha256_path(base_tokenizer_json)
    if observed_base_sha != expected_base_sha:
        raise ValueError("base tokenizer.json SHA-256 mismatch")
    expected_model_config_sha = _validate_sha256_literal(
        args.expected_model_config_sha256,
        "expected Mamba2 config.json SHA-256",
    )
    expected_checkpoint_sha = _validate_sha256_literal(
        args.expected_checkpoint_state_sha256,
        "expected DAPT checkpoint SHA-256",
    )

    tokenizer_binding = load_locked_augmented_tokenizer(
        resolve_path(args.augmented_tokenizer_dir),
        expected_tokenizer_json_sha256=expected_augmented_sha,
    )
    offset_contract = validate_locked_bytelevel_offset_contract(
        tokenizer_binding.tokenizer_json_path
    )
    tokenizer_binding, checkpoint_identity = _bind_tokenizer_to_locked_checkpoint(
        tokenizer_binding,
        model_dir=resolve_path(args.model_dir),
        checkpoint_dir=resolve_path(args.checkpoint_dir),
        expected_model_config_sha256=expected_model_config_sha,
        expected_checkpoint_state_sha256=expected_checkpoint_sha,
    )
    tokenizer_preflight = tokenizer_binding.preflight
    code_sha256 = _locked_code_hashes()

    examples = _read_train_examples(resolve_path(args.benchmark_root))
    frames = build_parser_v2_cache_frames(
        examples,
        tokenizer_binding=tokenizer_binding,
        split=IMPLEMENTATION_STAGE_SPLIT,
        parser_config=ParserV2Config(
            min_sections=2,
            max_content_tokens=511,
            max_heading_chars=80,
            max_heading_words=12,
        ),
        offset_contract_metadata=offset_contract,
    )
    frames.summary["base_tokenizer_json_sha256"] = observed_base_sha
    frames.summary["benchmark_columns_opened"] = "example_id|text"
    frames.summary["outcome_label_column_opened"] = False
    frames.summary["patient_identifier_column_opened"] = False
    frames.summary["offset_amendment_sha256"] = offset_amendment_sha256
    frames.summary.update(checkpoint_identity)

    return write_parser_v2_cache(
        frames,
        output_dir=output_dir,
        protocol_sha256=protocol_sha256,
        code_sha256=code_sha256,
        tokenizer_preflight=tokenizer_preflight,
        offset_amendment_sha256=offset_amendment_sha256,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build the restricted train-only Parser-v2 occurrence, span, "
            "direct-token, note-index, and fallback-chunk caches. Validation "
            "and test access are intentionally unavailable."
        )
    )
    parser.add_argument(
        "--benchmark-root",
        default=DEFAULT_BENCHMARK_ROOT,
        help="Locked LAMP-R root containing benchmark_examples_train.parquet.",
    )
    parser.add_argument(
        "--base-tokenizer-dir",
        required=True,
        help="Pinned local base-tokenizer snapshot directory.",
    )
    parser.add_argument(
        "--augmented-tokenizer-dir",
        required=True,
        help="Locked DAPT-checkpoint augmented tokenizer directory.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Pinned local benchang1110/mamba2-130m-hf snapshot directory.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Locked checkpoint-step-14763 directory.",
    )
    parser.add_argument(
        "--expected-base-tokenizer-json-sha256",
        default=LOCKED_BASE_TOKENIZER_JSON_SHA256,
    )
    parser.add_argument(
        "--expected-augmented-tokenizer-json-sha256",
        default=LOCKED_AUGMENTED_TOKENIZER_JSON_SHA256,
    )
    parser.add_argument(
        "--expected-model-config-sha256",
        default=LOCKED_MODEL_CONFIG_SHA256,
    )
    parser.add_argument(
        "--expected-checkpoint-state-sha256",
        default=LOCKED_CHECKPOINT_STATE_SHA256,
    )
    parser.add_argument("--protocol-path", default=DEFAULT_PROTOCOL)
    parser.add_argument("--protocol-sha256-path", default=None)
    parser.add_argument(
        "--offset-amendment-path",
        default=DEFAULT_OFFSET_AMENDMENT,
    )
    parser.add_argument("--offset-amendment-sha256-path", default=None)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    print(
        "PARSER_V2_CACHE_BUILD "
        "status=STARTED split=train validation_rows_opened=0 "
        "test_rows_opened=0 label_column_opened=false",
        flush=True,
    )
    result = run_builder(args)
    summary = result["summary"]
    print(
        "PARSER_V2_CACHE_BUILD "
        f"status={result['status']} "
        f"examples={summary['example_count']} "
        f"occurrences={summary['occurrence_count']} "
        f"model_slots={summary['model_slot_count']} "
        f"fallback_examples={summary['fallback_example_count']} "
        f"zero_slot_examples={summary['zero_model_slot_example_count']} "
        f"manifest_sha256={result['manifest_sha256']} "
        "validation_rows_opened=0 test_rows_opened=0",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
