# LAMP-R Benchmark Specification

LAMP-R is a leakage-aware MIMIC-IV-Note post-discharge readmission benchmark for retrospective discharge-summary risk modeling.

## Prediction Unit

The prediction unit is a merged continuous-care episode with a qualifying terminal discharge summary.

Admissions for the same patient are merged into one continuous-care episode when the next admission starts within the configured episode-merge gap after the previous discharge. The representative note is selected only from the terminal admission of that merged episode, identified by `end_hadm_id`. Intermediate transfer-admission discharge summaries are not used as representative benchmark notes.

If multiple qualifying terminal discharge summaries exist for the same `subject_id` and `episode_id`, the selected benchmark note is the earliest note by `availability_time`, with `note_time` and `note_id` used as deterministic tie-breakers. The benchmark keeps at most one labeled example per `subject_id` and `episode_id`.

## Note Eligibility

`note_time` is `storetime` when available, otherwise `charttime`.

A terminal discharge summary qualifies for the main benchmark only when:

- `note_time` is present.
- note text is non-empty.
- `note_time <= episode_dischtime + max_note_delay_hours`.

Episodes without a qualifying terminal discharge summary are excluded from the main benchmark.

## Availability Time

`availability_time = max(episode_dischtime, note_time)`.

This value is saved in benchmark outputs and is the leakage boundary for readmission labeling. A positive readmission must occur after `availability_time`.

## Label

The default binary label is 30-day same-system unplanned readmission after discharge-summary availability.

Positive examples require:

- first eligible unplanned readmission time `> availability_time`
- first eligible unplanned readmission time `<= episode_dischtime + horizon_days`

Planned admissions are not counted as positive under the default `continue_search` strategy; the search skips planned admissions and continues for the first unplanned admission within the horizon.

The priority order for exclusion and labeling is:

1. missing or empty terminal note
2. late terminal note
3. event before or at `availability_time`
4. death censoring
5. planned-first exclusion when configured
6. positive readmission within horizon
7. negative if no valid readmission and not censored

## Death Censoring

Post-discharge deaths before readmission and within the horizon are censored and excluded from the fixed-horizon binary readmission benchmark. If readmission occurs before death, the readmission label is retained.

## Splits

Splits are patient-level. No `subject_id` may overlap across train, validation, and test.

## Section Parsing

The section parser keeps occurrence-level parser output for diagnostics. Aggregate parser reports distinguish:

- `section_occurrence_count.<SECTION>`: total detected occurrences; this may exceed `n_examples`
- `section_note_count.<SECTION>`: number of examples with at least one occurrence
- `section_note_count_parsed.<SECTION>`: number of non-fallback parsed examples with at least one occurrence
- `section_note_prevalence_all.<SECTION>`: `section_note_count / n_examples`
- `section_note_prevalence_parsed.<SECTION>`: `section_note_count_parsed / parser_success_count`
- repeat-section metrics including `section_repeat_note_count`, `section_repeat_rate`, and `section_mean_occurrences_per_note_with_section`

For NoteSSM-v2 model input, repeated canonical sections within the same note are merged before section-aware pooling. Repeated section texts are concatenated in original order with `<SEC_REPEAT>`, so one canonical section produces at most one `<SEC_SUM>` embedding per note.

The builder-level `used_fallback` flag is a whitespace-token QC approximation. The model dataloader should recompute tokenizer-aware fallback using the actual tokenizer and merged canonical sections.

## DAPT

The strict main setting uses train patients only. Patient sequences are chronological and are packed without mixing patients in the same causal block.

`dapt_train_sequences.parquet` is a restricted local artifact because it contains note text sequences. `dapt_train_manifest.parquet` is a QC manifest; its block lengths are approximate unless regenerated with the final tokenizer.

## Release Boundary

Only code, configs, aggregate counts, metrics, and text-free reports are suitable for release. Raw clinical text and raw MIMIC-derived datasets must not be redistributed.

The no-text manifest must not contain raw note text, `sequence_text`, or raw `subject_id`.
