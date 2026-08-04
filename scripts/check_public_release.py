from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


FORBIDDEN_ROOT_DIRS = {
    ".git",
    "data",
    "outputs",
    "results",
    "figures",
    "logs",
    "checkpoints",
    "cache",
    "transfers",
    "provenance",
}

FORBIDDEN_DIR_NAMES = {
    "__pycache__",
    ".pytest_cache",
}

FORBIDDEN_SUFFIXES = {
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
    ".npy",
    ".npz",
    ".parquet",
    ".pkl",
    ".pickle",
    ".sqlite",
    ".db",
    ".tar",
    ".gz",
    ".zip",
    ".log",
}

SECRET_PATTERNS = [
    re.compile(r"HF_TOKEN\s*=", re.IGNORECASE),
    re.compile(r"HUGGINGFACE_TOKEN\s*=", re.IGNORECASE),
    re.compile(r"api[_-]?key\s*=", re.IGNORECASE),
    re.compile(r"password\s*=", re.IGNORECASE),
    re.compile(r"secret\s*=", re.IGNORECASE),
    re.compile(r"C:\\Users\\Lab", re.IGNORECASE),
    re.compile("/home/" + "mshin", re.IGNORECASE),
]

MAX_TEXT_FILE_BYTES = 5_000_000
MANIFEST_NAME = "PUBLIC_RELEASE_MANIFEST.txt"
MANIFEST_LINE_PATTERN = re.compile(r"^([0-9a-f]{64})  (.+)$")


def iter_files(root: Path):
    for path in root.rglob("*"):
        rel_parts = path.relative_to(root).parts
        if rel_parts and rel_parts[0] in FORBIDDEN_ROOT_DIRS:
            continue
        if set(rel_parts) & FORBIDDEN_DIR_NAMES:
            continue
        if path.is_file():
            yield path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_files(root: Path, files: list[Path]) -> list[Path]:
    return sorted(
        (path for path in files if path.relative_to(root).as_posix() != MANIFEST_NAME),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def write_release_manifest(root: Path, files: list[Path]) -> None:
    manifest_path = root / MANIFEST_NAME
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root).as_posix()}"
        for path in manifest_files(root, files)
    ]
    temporary_path = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    temporary_path.replace(manifest_path)


def validate_release_manifest(root: Path, files: list[Path]) -> list[str]:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.exists():
        return [f"Missing release manifest: {MANIFEST_NAME}"]

    failures: list[str] = []
    records: dict[str, str] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        match = MANIFEST_LINE_PATTERN.fullmatch(line)
        if match is None:
            failures.append(f"Malformed release manifest line {line_number}: {line!r}")
            continue
        expected_sha, relative_name = match.groups()
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            failures.append(f"Unsafe release manifest path on line {line_number}: {relative_name}")
            continue
        normalized_name = relative_path.as_posix()
        if normalized_name in records:
            failures.append(f"Duplicate release manifest entry: {normalized_name}")
            continue
        records[normalized_name] = expected_sha

    expected_paths = {
        path.relative_to(root).as_posix(): path
        for path in manifest_files(root, files)
    }
    for relative_name in sorted(set(expected_paths) - set(records)):
        failures.append(f"File missing from release manifest: {relative_name}")
    for relative_name in sorted(set(records) - set(expected_paths)):
        failures.append(f"Stale release manifest entry: {relative_name}")
    for relative_name in sorted(set(records) & set(expected_paths)):
        actual_sha = sha256_file(expected_paths[relative_name])
        if actual_sha != records[relative_name]:
            failures.append(
                f"Release manifest hash mismatch: {relative_name} "
                f"expected={records[relative_name]} actual={actual_sha}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check that a NoteSSM public release folder does not contain private artifacts.")
    parser.add_argument("root", nargs="?", default=".")
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=f"Regenerate {MANIFEST_NAME} after all privacy checks pass.",
    )
    args = parser.parse_args()

    root = Path(args.root).resolve()
    failures: list[str] = []

    for forbidden in sorted(FORBIDDEN_ROOT_DIRS):
        candidate = root / forbidden
        if candidate.exists() and forbidden != ".git":
            failures.append(f"Forbidden directory present: {candidate.relative_to(root)}")
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        rel_parts = path.relative_to(root).parts
        if set(rel_parts) & FORBIDDEN_DIR_NAMES:
            failures.append(f"Generated cache directory present: {path.relative_to(root)}")

    files = list(iter_files(root))
    for path in files:
        rel = path.relative_to(root)
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            if rel.parts[:1] != ("paper_tables",):
                failures.append(f"Forbidden artifact suffix: {rel}")
        if path.suffix.lower() == ".csv" and rel.parts[:1] != ("paper_tables",):
            failures.append(f"CSV outside paper_tables is not allowed: {rel}")
        if path.stat().st_size > MAX_TEXT_FILE_BYTES:
            failures.append(f"Large file over {MAX_TEXT_FILE_BYTES} bytes: {rel}")
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                failures.append(f"Potential private/secret pattern {pattern.pattern!r} in {rel}")

    if not failures and args.write_manifest:
        write_release_manifest(root, files)
        files = list(iter_files(root))

    if not failures:
        failures.extend(validate_release_manifest(root, files))

    if failures:
        print("PUBLIC_RELEASE_CHECK_FAILED")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if args.write_manifest:
        print(f"PUBLIC_RELEASE_MANIFEST_WRITTEN {root / MANIFEST_NAME}")
    print("PUBLIC_RELEASE_CHECK_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
