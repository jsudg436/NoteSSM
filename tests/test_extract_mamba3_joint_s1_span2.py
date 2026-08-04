from __future__ import annotations

import importlib.util
from argparse import Namespace
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "extract_mamba3_joint_s1_span2.py"
SPEC = importlib.util.spec_from_file_location("extract_mamba3_joint_s1_span2_under_test", SCRIPT_PATH)
extractor = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)

PARITY_PATH = ROOT / "scripts" / "diagnose_mamba3_joint_s1_parity.py"
PARITY_SPEC = importlib.util.spec_from_file_location("diagnose_mamba3_joint_s1_parity_under_test", PARITY_PATH)
parity = importlib.util.module_from_spec(PARITY_SPEC)
assert PARITY_SPEC.loader is not None
sys.modules[PARITY_SPEC.name] = parity
PARITY_SPEC.loader.exec_module(parity)


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = -1
    eos_token = "<EOS>"
    pad_token = "<PAD>"
    all_special_ids = [0, 1, 99]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<SEC_SUM>": 99, "<CHUNK_SUM>": 98}.get(token, self.unk_token_id)

    def __call__(self, texts, *, add_special_tokens=True, padding=False, truncation=True, max_length=None):
        rows = []
        for text in texts:
            ids = [10 + index for index, _ in enumerate(str(text).split())]
            if add_special_tokens:
                ids = [1] + ids
            if truncation and max_length is not None:
                ids = ids[:max_length]
            rows.append(ids)
        return {"input_ids": rows}


class CountingBackbone:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, input_ids):
        self.calls += 1
        batch, width = input_ids.shape
        row_ids = torch.arange(batch, device=input_ids.device).view(batch, 1).expand(batch, width)
        positions = torch.arange(width, device=input_ids.device).view(1, width).expand(batch, width)
        return torch.stack([input_ids.float(), positions.float(), row_ids.float()], dim=-1)


class FakeModel:
    def __init__(self) -> None:
        self.backbone = CountingBackbone()


def section_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "example_id": ["a", "b"],
            "split": ["train", "val"],
            "canonical_section": ["HPI", "PMH"],
            "section_order": [0, 1],
            "merged_section_text": ["one two three four", "solo"],
            "token_length": [4, 1],
            "use_chunk_fallback": [False, True],
        }
    )


def test_one_forward_generates_s1_and_s2() -> None:
    model = FakeModel()
    spec = extractor.extraction_spec("section")
    s1, s2, meta = extractor.extract_joint_batch(
        section_frame(),
        spec=spec,
        tokenizer=FakeTokenizer(),
        model=model,
        max_length=512,
        device=torch.device("cpu"),
    )
    assert model.backbone.calls == 1
    assert s1.shape == (2, 3)
    assert s2.shape == (3, 3)
    assert meta["evidence_slot"].tolist() == [0, 1, 0]


def test_span2_exact_ordered_coverage_and_short_inputs() -> None:
    spec = extractor.extraction_spec("section")
    _, _, meta = extractor.extract_joint_batch(
        section_frame(),
        spec=spec,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        max_length=512,
        device=torch.device("cpu"),
    )
    first = meta.iloc[:2]
    assert first["token_start_position"].tolist() == [1, 3]
    assert first["token_end_position"].tolist() == [3, 5]
    assert first["token_count"].tolist() == [2, 2]
    short = meta.iloc[2:]
    assert short["token_start_position"].tolist() == [1]
    assert short["token_end_position"].tolist() == [2]
    assert short["token_count"].tolist() == [1]


def test_metadata_order_is_deterministic_and_text_free() -> None:
    spec = extractor.extraction_spec("section")
    _, s2, span_rows = extractor.extract_joint_batch(
        section_frame(),
        spec=spec,
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        max_length=512,
        device=torch.device("cpu"),
    )
    metadata = extractor.build_span_metadata(section_frame(), span_rows, spec, shard_idx=0, output_offset=0)
    assert len(metadata) == len(s2)
    assert metadata["example_id"].tolist() == ["a", "a", "b"]
    assert metadata["representation"].tolist() == ["span_mean_2"] * 3
    assert not (extractor.FORBIDDEN_METADATA & set(metadata.columns))
    assert "merged_section_text" not in metadata.columns


def test_write_s1_stores_marker_and_span2_prefixes_from_one_pass() -> None:
    with TemporaryDirectory() as tmpdir:
        model = FakeModel()
        args = Namespace(
            output_dir=tmpdir,
            section_cache_dir=tmpdir,
            shard_size=2,
            resume=False,
            s1_cache_dir="",
            no_s1_audit=True,
            max_length=512,
            batch_size=2,
            split="train_val",
            audit_max_abs_tol=1e-3,
            audit_min_cosine=0.99999,
            source_bundle_manifest_sha256="bundle" * 13,
            source_kind_manifest_sha256="kind" * 16,
            source_manifest_sha256="kind" * 16,
            model_id="state-spaces/mamba3-siso-187m",
            tokenizer_id="meta-llama/Llama-3.1-8B",
            dtype="bf16",
            write_s1=True,
            s2_prefix_suffix="span2",
        )
        extractor.extract_to_shards(
            section_frame(),
            spec=extractor.extraction_spec("section"),
            tokenizer=FakeTokenizer(),
            model=model,
            args=args,
            checkpoint_info={"sha256": "a" * 64, "global_step": 25338},
            tokenizer_info=tokenizer_info(),
            device=torch.device("cpu"),
        )

        root = Path(tmpdir)
        assert model.backbone.calls == 1
        s1 = np.load(root / "section_embeddings_shard_000.npy")
        s2 = np.load(root / "section_span2_embeddings_shard_000.npy")
        s1_meta = pd.read_parquet(root / "section_embedding_metadata_shard_000.parquet")
        s2_meta = pd.read_parquet(root / "section_span2_embedding_metadata_shard_000.parquet")
        assert s1.shape == (2, 3)
        assert s2.shape == (3, 3)
        assert s1_meta["representation"].tolist() == ["sec_sum_1"] * 2
        assert s2_meta["representation"].tolist() == ["span_mean_2"] * 3
        complete_path = root / "section_joint_embedding_shard_000.complete.json"
        assert complete_path.exists()
        assert not (root / "section_embedding_shard_000.complete.json").exists()
        complete = extractor.json.loads(complete_path.read_text())
        assert complete["schema_version"] == 5
        assert complete["input_kind"] == "section"
        assert complete["s1_prefix"] == "section"
        assert complete["s2_prefix"] == "section_span2"
        assert complete["s1_rows"] == 2
        assert complete["s2_rows"] == 3
        assert complete["s1_embeddings_sha256"] == extractor.sha256_file(root / "section_embeddings_shard_000.npy")
        assert complete["s1_metadata_sha256"] == extractor.sha256_file(root / "section_embedding_metadata_shard_000.parquet")
        assert complete["s2_embeddings_sha256"] == extractor.sha256_file(root / "section_span2_embeddings_shard_000.npy")
        assert complete["s2_metadata_sha256"] == extractor.sha256_file(root / "section_span2_embedding_metadata_shard_000.parquet")
        assert complete["raw_text_stored"] is False
        assert complete["test_accessed"] is False
        assert "forward_contract_sha256" in complete
        run_id = complete["joint_run_id"]
        assert (root / "section_embedding_shards_summary.csv").exists()
        assert (root / "section_span2_embedding_shards_summary.csv").exists()

        args.resume = True
        extractor.extract_to_shards(
            section_frame(),
            spec=extractor.extraction_spec("section"),
            tokenizer=FakeTokenizer(),
            model=model,
            args=args,
            checkpoint_info={"sha256": "a" * 64, "global_step": 25338},
            tokenizer_info=tokenizer_info(),
            device=torch.device("cpu"),
        )
        assert model.backbone.calls == 1
        resumed_complete = extractor.json.loads(complete_path.read_text())
        assert resumed_complete["joint_run_id"] == run_id


def test_resume_rejects_partial_shard() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        s1_paths = extractor.shard_data_paths(root, "section", 0)
        s2_paths = extractor.shard_data_paths(root, "section_span2", 0)
        complete_path = extractor.joint_completion_path(root, "section", 0)
        np.save(s1_paths["embeddings"], np.zeros((1, 3), dtype=np.float32))
        with pytest.raises(ValueError, match="Partial joint shard"):
            extractor.check_joint_resume_state(s1_paths, s2_paths, complete_path, resume=True)


def test_resume_moves_partial_joint_shard_to_stale_and_recomputes() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        s1_paths = extractor.shard_data_paths(root, "section", 0)
        s2_paths = extractor.shard_data_paths(root, "section_span2", 0)
        complete_path = extractor.joint_completion_path(root, "section", 0)
        np.save(s1_paths["embeddings"], np.zeros((1, 3), dtype=np.float32))

        state = extractor.check_joint_resume_state(
            s1_paths,
            s2_paths,
            complete_path,
            resume=True,
            stale_label="section_joint_shard_000",
        )

        assert state == "missing"
        assert not s1_paths["embeddings"].exists()
        stale_files = list((root / "_stale_partial_joint_shards").glob("section_joint_shard_000_*/*"))
        assert stale_files


def completion_args() -> Namespace:
    return Namespace(
        max_length=512,
        dtype="bf16",
        batch_size=64,
        section_cache_dir=".",
        split="train_val",
        limit_examples=None,
        limit_rows=None,
        source_manifest_sha256="source" * 16,
        source_bundle_manifest_sha256="bundle" * 13,
        source_kind_manifest_sha256="source" * 16,
        model_id="state-spaces/mamba3-siso-187m",
        tokenizer_id="meta-llama/Llama-3.1-8B",
        audit_max_abs_tol=1e-3,
        audit_min_cosine=0.99999,
        write_s1=True,
        s2_prefix_suffix="span2",
    )


def tokenizer_info() -> dict[str, object]:
    return {
        "tokenizer_fingerprint": "t" * 64,
        "tokenizer_run_fingerprint": "r" * 64,
        "tokenizer_vocab_size": 128258,
        "tokenizer_backend_sha256": "b" * 64,
        "tokenizer_commit_hash": "revision",
        "marker_id": 128256,
    }


def test_source_digest_locks_text_and_safe_metadata_without_storing_text() -> None:
    spec = extractor.extraction_spec("section")
    original = section_frame()
    text_changed = original.copy()
    text_changed.loc[0, "merged_section_text"] = "changed content"
    metadata_changed = original.copy()
    metadata_changed.loc[0, "token_length"] = 999

    digest = extractor.shard_source_key_digest(original, spec)
    assert digest != extractor.shard_source_key_digest(text_changed, spec)
    assert digest != extractor.shard_source_key_digest(metadata_changed, spec)


def test_source_manifests_use_fallback_chunk_cache_and_bundle_run_id() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        section = section_frame()
        chunk = pd.DataFrame(
            {
                "example_id": ["a"],
                "split": ["train"],
                "chunk_order": [0],
                "chunk_text": ["chunk text"],
                "token_length": [2],
            }
        )
        for split in ["train", "val"]:
            section.to_parquet(root / f"section_cache_{split}.parquet", index=False)
            chunk.to_parquet(root / f"fallback_chunk_cache_{split}.parquet", index=False)

        section_kind = extractor.source_kind_cache_file_manifest_fingerprint(root, "section", "train_val")
        chunk_kind = extractor.source_kind_cache_file_manifest_fingerprint(root, "chunk", "train_val")
        bundle = extractor.source_bundle_cache_file_manifest_fingerprint(root, "train_val")

        assert section_kind != chunk_kind
        assert bundle not in {section_kind, chunk_kind}

        base_args = completion_args()
        base_args.section_cache_dir = str(root)
        base_args.source_bundle_manifest_sha256 = bundle
        base_args.source_kind_manifest_sha256 = section_kind
        section_run_id = extractor.deterministic_joint_run_id(
            args=base_args,
            checkpoint_info={"sha256": "a" * 64, "global_step": 25338},
            tokenizer_info=tokenizer_info(),
        )
        chunk_args = completion_args()
        chunk_args.section_cache_dir = str(root)
        chunk_args.source_bundle_manifest_sha256 = bundle
        chunk_args.source_kind_manifest_sha256 = chunk_kind
        chunk_run_id = extractor.deterministic_joint_run_id(
            args=chunk_args,
            checkpoint_info={"sha256": "a" * 64, "global_step": 25338},
            tokenizer_info=tokenizer_info(),
        )
        assert section_run_id == chunk_run_id

        chunk_changed = chunk.copy()
        chunk_changed.loc[0, "chunk_text"] = "changed"
        chunk_changed.to_parquet(root / "fallback_chunk_cache_val.parquet", index=False)
        changed_bundle = extractor.source_bundle_cache_file_manifest_fingerprint(root, "train_val")
        assert changed_bundle != bundle


def test_compare_sharded_embedding_dirs_checks_every_shard() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        left = root / "left"
        right = root / "right"
        left.mkdir()
        right.mkdir()
        for shard_idx in [0, 1]:
            array = np.full((2, 3), shard_idx, dtype=np.float32)
            metadata = pd.DataFrame({"example_id": [f"e{shard_idx}a", f"e{shard_idx}b"], "row_index": [0, 1]})
            for side in [left, right]:
                np.save(side / f"section_embeddings_shard_{shard_idx:03d}.npy", array)
                metadata.to_parquet(side / f"section_embedding_metadata_shard_{shard_idx:03d}.parquet", index=False)

        result = extractor.compare_sharded_embedding_dirs(left, right, "section")
        assert result["n_shards"] == 2
        assert result["n_rows"] == 4

        changed = np.load(right / "section_embeddings_shard_001.npy")
        changed[0, 0] += 1e-3
        np.save(right / "section_embeddings_shard_001.npy", changed)
        with pytest.raises(ValueError, match="shard_001"):
            extractor.compare_sharded_embedding_dirs(left, right, "section")


def test_parity_diagnostic_records_old_new_s1_difference_without_hard_gate() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        old_root = root / "old"
        joint_root = root / "joint"
        old_root.mkdir()
        joint_root.mkdir()
        metadata = pd.DataFrame(
            {
                "example_id": ["a", "b"],
                "section_order": [0, 1],
                "row_index": [0, 1],
            }
        )
        old_vectors = np.ones((2, 3), dtype=np.float32)
        joint_vectors = old_vectors.copy()
        joint_vectors[0, 0] += 0.01
        np.save(old_root / "section_embeddings_shard_000.npy", old_vectors)
        metadata.to_parquet(old_root / "section_embedding_metadata_shard_000.parquet", index=False)
        np.save(joint_root / "section_embeddings_shard_000.npy", joint_vectors)
        metadata.to_parquet(joint_root / "section_embedding_metadata_shard_000.parquet", index=False)

        row = parity.compare_prefix(
            old_root=old_root,
            joint_root=joint_root,
            input_kind="section",
            max_abs_tolerance=1e-3,
            minimum_cosine_tolerance=0.99999,
        )

        assert row["status"] == "ok"
        assert row["n_expected"] == 2
        assert row["n_aligned"] == 2
        assert row["n_missing"] == 0
        assert row["max_abs_diff"] > 1e-3
        assert row["within_tolerance"] is False
        assert not (extractor.FORBIDDEN_METADATA & set(row))


def test_completed_resume_schema_mismatch_fails_before_reuse() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = extractor.extraction_spec("section")
        shard = section_frame()
        s1_paths = extractor.shard_data_paths(root, "section", 0)
        s2_paths = extractor.shard_data_paths(root, "section_span2", 0)
        complete_path = extractor.joint_completion_path(root, "section", 0)
        s1_metadata = pd.DataFrame({"example_id": ["a", "b"], "row_index": [0, 1]})
        s2_metadata = pd.DataFrame({"example_id": ["a", "a", "b"], "row_index": [0, 1, 2]})
        np.save(s1_paths["embeddings"], np.zeros((2, 3), dtype=np.float32))
        s1_metadata.to_parquet(s1_paths["metadata"], index=False)
        np.save(s2_paths["embeddings"], np.zeros((3, 3), dtype=np.float32))
        s2_metadata.to_parquet(s2_paths["metadata"], index=False)
        args = completion_args()
        checkpoint_info = {"sha256": "a" * 64, "global_step": 25338}
        expected = extractor.expected_joint_completion_fields(
            spec=spec,
            shard_idx=0,
            shard=shard,
            args=args,
            checkpoint_info=checkpoint_info,
            tokenizer_info=tokenizer_info(),
            audit_enabled=True,
            s1_cache_fingerprint="s1" * 32,
        )
        completion = {
            **expected,
            "schema_version": 2,
            "s1_rows": len(s1_metadata),
            "s2_rows": len(s2_metadata),
            "s1_embeddings_sha256": extractor.sha256_file(s1_paths["embeddings"]),
            "s1_metadata_sha256": extractor.sha256_file(s1_paths["metadata"]),
            "s2_embeddings_sha256": extractor.sha256_file(s2_paths["embeddings"]),
            "s2_metadata_sha256": extractor.sha256_file(s2_paths["metadata"]),
            "s1_audit": extractor.S1AuditStats().to_dict(),
        }
        extractor.write_completion(complete_path, completion)

        with pytest.raises(ValueError, match="schema_version"):
            extractor.load_completed_joint_metadata(
                s1_paths=s1_paths,
                s2_paths=s2_paths,
                complete_path=complete_path,
                expected=expected,
                require_audit=True,
            )


def test_completed_resume_verifies_hashes_and_restores_audit() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        spec = extractor.extraction_spec("section")
        shard = section_frame()
        s1_paths = extractor.shard_data_paths(root, "section", 0)
        s2_paths = extractor.shard_data_paths(root, "section_span2", 0)
        complete_path = extractor.joint_completion_path(root, "section", 0)
        s1_metadata = pd.DataFrame({"example_id": ["a", "b"], "row_index": [0, 1]})
        s2_metadata = pd.DataFrame({"example_id": ["a", "a", "b"], "row_index": [0, 1, 2]})
        np.save(s1_paths["embeddings"], np.zeros((2, 3), dtype=np.float32))
        s1_metadata.to_parquet(s1_paths["metadata"], index=False)
        np.save(s2_paths["embeddings"], np.zeros((3, 3), dtype=np.float32))
        s2_metadata.to_parquet(s2_paths["metadata"], index=False)
        args = completion_args()
        checkpoint_info = {"sha256": "a" * 64, "global_step": 25338}
        expected = extractor.expected_joint_completion_fields(
            spec=spec,
            shard_idx=0,
            shard=shard,
            args=args,
            checkpoint_info=checkpoint_info,
            tokenizer_info=tokenizer_info(),
            audit_enabled=True,
            s1_cache_fingerprint="s1" * 32,
        )
        audit = extractor.S1AuditStats()
        audit.n_compared = 2
        audit.n_values = 6
        audit.sum_abs_diff = 0.001
        audit.max_abs_diff = 0.001
        audit.min_cosine = 0.99999
        audit.sum_cosine = 1.99999
        completion = {
            **expected,
            "s1_rows": len(s1_metadata),
            "s2_rows": len(s2_metadata),
            "s1_embeddings_sha256": extractor.sha256_file(s1_paths["embeddings"]),
            "s1_metadata_sha256": extractor.sha256_file(s1_paths["metadata"]),
            "s2_embeddings_sha256": extractor.sha256_file(s2_paths["embeddings"]),
            "s2_metadata_sha256": extractor.sha256_file(s2_paths["metadata"]),
            "s1_audit": audit.to_dict(),
        }
        extractor.write_completion(complete_path, completion)

        loaded_s1, loaded_s2, loaded_completion = extractor.load_completed_joint_metadata(
            s1_paths=s1_paths,
            s2_paths=s2_paths,
            complete_path=complete_path,
            expected=expected,
            require_audit=True,
        )
        restored = extractor.S1AuditStats.from_dict(loaded_completion["s1_audit"])
        assert loaded_s1 is not None
        assert len(loaded_s1) == 2
        assert len(loaded_s2) == 3
        assert restored.n_compared == 2

        completion["s2_metadata_sha256"] = "b" * 64
        extractor.write_completion(complete_path, completion)
        with pytest.raises(ValueError, match="S2 metadata SHA256 mismatch"):
            extractor.load_completed_joint_metadata(
                s1_paths=s1_paths,
                s2_paths=s2_paths,
                complete_path=complete_path,
                expected=expected,
                require_audit=True,
            )

        completion["s2_metadata_sha256"] = extractor.sha256_file(s2_paths["metadata"])
        extractor.write_completion(complete_path, completion)
        changed_s1 = np.load(s1_paths["embeddings"])
        changed_s1[0, 0] = 1.0
        np.save(s1_paths["embeddings"], changed_s1)
        with pytest.raises(ValueError, match="S1 embedding SHA256 mismatch"):
            extractor.load_completed_joint_metadata(
                s1_paths=s1_paths,
                s2_paths=s2_paths,
                complete_path=complete_path,
                expected=expected,
                require_audit=True,
            )


def test_completion_contract_locks_tokenizer_tolerance_and_s1_cache() -> None:
    spec = extractor.extraction_spec("section")
    shard = section_frame()
    checkpoint_info = {"sha256": "a" * 64, "global_step": 25338}
    expected = extractor.expected_joint_completion_fields(
        spec=spec,
        shard_idx=0,
        shard=shard,
        args=completion_args(),
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info(),
        audit_enabled=True,
        s1_cache_fingerprint="s1" * 32,
    )

    changed_args = completion_args()
    changed_args.audit_max_abs_tol = 5e-2
    changed_tolerance = extractor.expected_joint_completion_fields(
        spec=spec,
        shard_idx=0,
        shard=shard,
        args=changed_args,
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info(),
        audit_enabled=True,
        s1_cache_fingerprint="s1" * 32,
    )
    changed_tokenizer = dict(tokenizer_info(), tokenizer_fingerprint="u" * 64)
    changed_tokenizer_expected = extractor.expected_joint_completion_fields(
        spec=spec,
        shard_idx=0,
        shard=shard,
        args=completion_args(),
        checkpoint_info=checkpoint_info,
        tokenizer_info=changed_tokenizer,
        audit_enabled=True,
        s1_cache_fingerprint="s1" * 32,
    )
    changed_run_tokenizer = dict(tokenizer_info(), tokenizer_run_fingerprint="v" * 64)
    changed_run_tokenizer_expected = extractor.expected_joint_completion_fields(
        spec=spec,
        shard_idx=0,
        shard=shard,
        args=completion_args(),
        checkpoint_info=checkpoint_info,
        tokenizer_info=changed_run_tokenizer,
        audit_enabled=True,
        s1_cache_fingerprint="s1" * 32,
    )
    changed_s1 = extractor.expected_joint_completion_fields(
        spec=spec,
        shard_idx=0,
        shard=shard,
        args=completion_args(),
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info(),
        audit_enabled=True,
        s1_cache_fingerprint="s2" * 32,
    )
    changed_source_args = completion_args()
    changed_source_args.source_bundle_manifest_sha256 = "changed_bundle" * 5
    changed_source_args.source_kind_manifest_sha256 = completion_args().source_kind_manifest_sha256
    changed_source = extractor.expected_joint_completion_fields(
        spec=spec,
        shard_idx=0,
        shard=shard,
        args=changed_source_args,
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info(),
        audit_enabled=True,
        s1_cache_fingerprint="s1" * 32,
    )
    changed_kind_args = completion_args()
    changed_kind_args.source_kind_manifest_sha256 = "changed_kind" * 6
    changed_kind = extractor.expected_joint_completion_fields(
        spec=spec,
        shard_idx=0,
        shard=shard,
        args=changed_kind_args,
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info(),
        audit_enabled=True,
        s1_cache_fingerprint="s1" * 32,
    )
    changed_forward_args = completion_args()
    changed_forward_args.max_length = 256
    changed_forward = extractor.expected_joint_completion_fields(
        spec=spec,
        shard_idx=0,
        shard=shard,
        args=changed_forward_args,
        checkpoint_info=checkpoint_info,
        tokenizer_info=tokenizer_info(),
        audit_enabled=True,
        s1_cache_fingerprint="s1" * 32,
    )

    assert expected["schema_version"] == 5
    assert expected["source_digest_sha256"] == expected["source_key_sha256"]
    assert expected["audit_max_abs_tol"] != changed_tolerance["audit_max_abs_tol"]
    assert expected["tokenizer_fingerprint"] != changed_tokenizer_expected["tokenizer_fingerprint"]
    assert expected["joint_run_id"] == changed_tokenizer_expected["joint_run_id"]
    assert expected["joint_run_id"] != changed_run_tokenizer_expected["joint_run_id"]
    assert expected["s1_cache_fingerprint"] != changed_s1["s1_cache_fingerprint"]
    assert expected["source_bundle_manifest_sha256"] != changed_source["source_bundle_manifest_sha256"]
    assert expected["joint_run_id"] != changed_source["joint_run_id"]
    assert expected["source_kind_manifest_sha256"] != changed_kind["source_kind_manifest_sha256"]
    assert expected["source_manifest_sha256"] == expected["source_kind_manifest_sha256"]
    assert expected["forward_contract_sha256"] != changed_forward["forward_contract_sha256"]


def test_train_validation_only_rejects_test() -> None:
    with pytest.raises(ValueError, match="train/validation only"):
        extractor.validate_split("test")
    with pytest.raises(ValueError, match="train/validation only"):
        extractor.validate_split("all")


def test_s1_audit_detects_alignment_mismatch() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        np.save(root / "section_embeddings_shard_000.npy", np.ones((1, 3), dtype=np.float32))
        pd.DataFrame(
            {
                "example_id": ["other"],
                "section_order": [0],
                "row_index_in_shard": [0],
            }
        ).to_parquet(root / "section_embedding_metadata_shard_000.parquet", index=False)
        audit = extractor.S1AuditIndex(root, "section", ["example_id", "section_order"])
        with pytest.raises(ValueError, match="could not align"):
            audit.vectors_for(pd.DataFrame({"example_id": ["a"], "section_order": [0]}))


def test_read_source_cache_does_not_modify_frozen_source() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        frame = section_frame()
        frame.to_parquet(root / "section_cache_train.parquet", index=False)
        frame.iloc[:1].to_parquet(root / "section_cache_val.parquet", index=False)
        before = (root / "section_cache_train.parquet").stat().st_mtime_ns
        out = extractor.read_source_cache(root, "section", "train_val", limit_examples=1)
        after = (root / "section_cache_train.parquet").stat().st_mtime_ns
        assert not out.empty
        assert before == after
