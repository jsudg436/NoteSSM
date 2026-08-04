# Public Release Checklist

Before pushing to GitHub:

- [ ] Confirm the `LICENSE` choice and copyright holder string.
- [ ] Confirm no raw clinical text is present.
- [ ] Confirm no MIMIC-IV tables or derived raw-text files are present.
- [ ] Confirm no checkpoints, embedding caches, token shards, prediction CSVs, or large artifacts are present.
- [ ] Confirm no Hugging Face tokens, API keys, `.env`, or machine-local runtime files are present.
- [ ] Regenerate the release manifest with `python scripts/check_public_release.py . --write-manifest`.
- [ ] Verify the release with `python scripts/check_public_release.py .`.
- [ ] Run `python -m pytest -q tests` in a clean environment.
- [ ] Run `bash -n scripts/*.sh` on Linux or a shell-capable CI runner.
- [ ] Verify the final result artifact is archived outside git with `FREEZE_MANIFEST.json` and `SHA256SUMS.txt`.
- [ ] Update README links to the paper, dataset access instructions, and external artifact DOI once available.

Recommended GitHub contents:

- Source code and tests.
- Safe configs and templates.
- Small aggregate tables.
- Protocol documentation.
- No protected clinical data or large model artifacts.
