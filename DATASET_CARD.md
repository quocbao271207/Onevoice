# OneVoice locked training/evaluation data

> Snapshot: 22/09/2026 (Asia/Bangkok). This is a research/non-commercial dataset assembly. It is not a redistributable dataset release.

## Intended use

The manifests support a cascaded Vietnamese↔English medical speech prototype:

- Vietnamese medical/code-switch ASR fine-tuning;
- English medical ASR evaluation only;
- bidirectional English↔Vietnamese medical MT fine-tuning and locked evaluation;
- general Vietnamese replay for acoustic robustness without replacing the medical objective.

Do not use these artifacts for autonomous diagnosis, treatment decisions, commercial release, or redistribution of upstream audio/text. Runtime output remains subject to number, unit, negation and terminology safety checks.

## Frozen upstream revisions

Exact revisions are machine-readable in `configs/artifact_lock.yaml`.

| Source | Revision | Role | License/constraint |
|---|---|---|---|
| `leduckhai/VietMed` | `cc7980cd...` | VI medical ASR | MIT |
| `tensorxt/ViMedCSS` | `b6959a18...` | VI/EN code-switch ASR | CC-BY-4.0; source YouTube audio is not redistributed |
| `duymanh1606/vietnamese_asr` (VIVOS mirror) | `851c7ed3...` | speaker-balanced VI replay | CC-BY-NC-SA-4.0 |
| `ekacare/eka-medical-asr-evaluation-dataset` | `433e6603...` | EN medical ASR test only | MIT; never used for training |
| `nhuvo/MedEV` | `6afaf6b4...` | EN↔VI medical MT | upstream card has no declared license; owner-approved research-only/no-redistribution |

## EDA and human review

- VietMed: 9,207 clips / 15.929 h; explicit accent, role, gender, ICD-10 and recording-condition fields.
- ViMedCSS: 15,818 clips / 32.643 h; 1,413 recording-video groups crossed the published roles, so the primary split was rebuilt group-disjoint.
- VIVOS: 11,660 train + 760 calibration clips; 46/19 speakers and no speaker/group overlap. Repeated scripted text from different speakers is retained.
- Eka: 3,619 evaluation-only clips / 8.401 h.
- MedEV: 717,592 raw rows = 358,796 aligned pairs. The actual layout is all English rows followed by all Vietnamese rows in each split; pairing is `i ↔ i + split_size/2`, not alternating rows.
- The owner listened to 36 core clips (12 per source) and a separate 12-clip VIVOS pack. All 48/48 retained samples were confirmed audio-clean with exact transcripts and no observed PII.
- The VIVOS decision is recorded in `data/review/vivos_listening_pack/review.csv` and `data/reports/eda/vivos.json`.

## Split and sampling policy

- Locked test material is processed before validation/train so lower-priority roles cannot capture a speaker, recording group or exact MT pair.
- VietMed keeps its official train/dev/test roles; its overlapping `cv` subset is calibration-only.
- ViMedCSS `hard` and every related video group are locked to test. Remaining train/validation/test video groups are deterministically hashed 85/7.5/7.5 with seed `20260922`.
- VIVOS contributes at most 80 train utterances per speaker (3,680 records before signal QC). Its official test is calibration-only and is not part of the medical headline score.
- Eka stays test-only.
- MedEV keeps official roles after exact-pair deduplication and higher-role precedence.

## Quality gates

Metadata/text gates remove empty text, invalid duration, sub-0.5 s audio, over-30 s audio, extreme text/audio ratio, MT language/identity errors, MT length ratio below 0.25 or above 4, and MT pairs over 2,000 characters. `number_set_mismatch` is retained as a review slice because punctuation and explanatory translation make it too noisy for automatic rejection.

PII/contact and evidence-backed corrupt IDs are versioned in `configs/data_exclusions.yaml`. Twelve ViMedCSS rows whose upstream audio asset stayed empty after four retries are excluded. Scientific DOI/SNP/table-number false positives are retained.

All downloaded audio is decoded before GPU use. Passing audio is stored losslessly as mono 16 kHz PCM16 FLAC. Fatal signal criteria are:

- decoded duration below 0.5 s;
- RMS below 0.003;
- clipped fraction above 20%;
- metadata duration difference above both 1.1 s and 20% (0.5–1.1 s remains a warning because ViMedCSS boundaries are integer-rounded);
- decode, non-finite or empty-signal failure.

## Final signal-QC manifests

Primary ASR:

| Role | Records | Known-duration hours | Composition |
|---|---:|---:|---|
| train | 17,806 | 31.860 h | 2,773 VietMed / 4.799h; 11,354 ViMedCSS / 22.132h; 3,679 VIVOS / 4.929h |
| validation | 3,733 | 6.619 h | 2,764 VietMed / 4.700h; 969 ViMedCSS / 1.920h |
| test | 10,390 | 20.912 h | 3,436 VietMed / 6.009h; 3,346 ViMedCSS / 6.518h; 3,608 Eka / 8.386h |

MT:

| Role | Pairs |
|---|---:|
| train | 339,028 |
| validation | 8,929 |
| test | 8,938 |

Signal QC rejected 69/17,918 train, 5/3,738 validation and 6/10,399 test rows. Reasons were 77 near-silent clips and three decoded text/audio-ratio failures (one also had a large duration mismatch). A subsequent exact-FLAC SHA-256 audit found 46 duplicate groups, including 17 crossing model roles despite different IDs/video groups; the pipeline retained the highest role (`test > validation > train`) and removed 43 train plus 3 duplicate-test rows. Final local manifests therefore contain 31,929 unique audio hashes, zero missing/absolute paths and zero duplicate ID, audio hash, speaker or recording-group overlap. Systematic one-second ViMedCSS metadata rounding remains a warning rather than being mislabeled as corrupt audio. Counts, hours and SHA-256 values are in `data/reports/local_audio_summary.json`; exact-audio removals are in `data/processed/manifests/asr--exact-audio-rejections.jsonl`.

## Known limitations

- Accent labels are available for VietMed but not ViMedCSS or VIVOS. The Vietnamese base model was already pretrained/fine-tuned on broad Vietnamese speech, but the final model still needs accent-sliced evaluation.
- Bud500 and VietSpeech were gated/inaccessible during this run.
- ViMD offers 63-province coverage but the full source is about 59.8 GB under CC-BY-NC-ND-4.0; it is reserved for a future legally reviewed accent benchmark, not training here.
- VIVOS is clean scripted speech and can improve general robustness but does not represent clinical conversation; its speaker cap prevents it from dominating.
- Eka contains only an evaluation split, so OneVoice does not claim a fine-tuned English ASR model.
- MedEV license ambiguity and NLLB's non-commercial license block commercial deployment.

## Reproduction

Run the commands under “Runbook” in `project_analysis.md`. The authoritative artifacts are:

- `configs/datasets.yaml`
- `configs/data_exclusions.yaml`
- `configs/artifact_lock.yaml`
- `data/reports/eda/`
- `data/reports/manifests/validation.json`
- `data/reports/audio_qc.json`
- `data/reports/audio_inventory.jsonl`
- `data/processed/manifests/`

Any data-policy, exclusion or signal-QC change requires re-merge, re-validation, new checksums and a new snapshot date.
