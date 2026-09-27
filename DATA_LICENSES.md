# OneVoice data and model use register

> Snapshot: 22/09/2026. Re-check upstream cards before redistribution or commercial use.

| Asset | Intended use | Declared license | OneVoice restriction |
|---|---|---|---|
| `leduckhai/VietMed` | VI medical ASR train/eval | MIT | Preserve attribution and locked test. |
| `tensorxt/ViMedCSS` | VI/EN code-switch ASR train/eval | CC-BY-4.0 | Attribute dataset; do not redistribute source YouTube audio until rights review. |
| `nhuvo/MedEV` | EN↔VI medical MT train/eval | Not declared on card | Research prototype only; no redistribution/commercial release until written clarification. |
| `ekacare/eka-medical-asr-evaluation-dataset` | EN medical ASR evaluation only | MIT | Never train on the test-only split. |
| `duymanh1606/vietnamese_asr` (official VIVOS mirror) | Speaker-balanced VI general replay/calibration | CC-BY-NC-SA-4.0 | Research/non-commercial only; preserve attribution/ShareAlike obligations; never report calibration as medical test. |
| `nguyendv02/ViMD_Dataset` | Future 63-province accent benchmark | CC-BY-NC-ND-4.0 | Not used for training in this run; legal review required before treating a trained model as a permitted derivative. |
| `linhtran92/viet_bud500` | Optional multi-accent robustness | CC-BY-NC-SA | Disabled; non-commercial/ShareAlike constraints require a separate decision. |
| `NhutP/VietSpeech` | Optional multi-accent robustness | Apache-2.0 shown on card | Disabled and gated; accept upstream access terms before use. |
| `minhtien2405/fptu-vovinam-dataset` | Candidate Central-heavy accent replay | MIT shown on mirror card | Disabled: extreme repeated-text rate, speaker leakage across published splits, and provenance/consent need verification. |
| `anonymous-vivoice34/ViVoice34` | Candidate province-labelled preview | CC-BY-4.0 shown on card | Disabled: anonymous preview, long unsegmented clips and separate large package need provenance review. |
| `vnpost-ai/govvox-100h-v3` | Candidate province/dialect benchmark | Other/manual access | Disabled until access and written license terms are approved. |
| `vinai/PhoWhisper-small` | VI ASR base model | Check upstream snapshot | Record exact revision in every experiment manifest. |
| `distil-whisper/distil-small.en` | EN ASR baseline | Apache-2.0 | English only; do not use for Vietnamese. |
| `facebook/nllb-200-distilled-600M` | MT base model | CC-BY-NC-4.0 | Research/non-commercial path only unless replaced with a commercially compatible base. |
| `rhasspy/piper-voices` voice artifacts | Offline TTS | Voice-specific model cards | Bundle only after reviewing each voice card/attribution requirement. |

Human review artifacts live under `data/review/` and are not committed because they can contain copied transcripts, paths, or reviewer notes. Dataset/model versions, decisions, and manifest SHA-256 values must be frozen in an experiment manifest before a GPU run.
