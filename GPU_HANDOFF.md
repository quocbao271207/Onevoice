# OneVoice cloud GPU handoff

> Prepared 22/09/2026. Human QA is complete (48/48 keep). Start only after `python scripts/preflight_project.py --gate gpu` returns `ready: true` on the rented machine.

## Recommended instance

Start with one NVIDIA A40 or RTX A6000 48 GB, at least 8 vCPU, 50 GB RAM and a 150 GB persistent volume. Use FP16, per-device batch 4 and gradient accumulation 8 to preserve effective batch 32. A100 80 GB is now only the optional fast path, not the default.

If no 48 GB card is available, a 24 GB RTX 3090, RTX 4090 or RTX A5000 can run the mandatory dry-run with per-device batch 2, validation batch 2 and gradient accumulation 16. Do not start a full run until its measured throughput and cost projection are acceptable.

Price snapshot checked 25/09/2026:

- RunPod Community Cloud lists A40 48 GB around USD 0.35/hour and RTX A6000 48 GB around USD 0.33/hour.
- Lower-cost 24 GB listings include RTX A5000 around USD 0.16/hour, RTX 3090 around USD 0.22/hour and RTX 4090 around USD 0.34/hour.
- Secure Cloud is more expensive but less interruptible: A40 around USD 0.49/hour and RTX A6000 around USD 0.53/hour, versus A100 80 GB around USD 1.59/hour.
- Source: <https://www.runpod.io/gpu-models> and <https://www.runpod.io/pricing>.

Prices and availability change. Do not prepay a long reservation. Set a USD 10 hard budget alert for the first two dry-runs, use persistent storage, and terminate the compute instance immediately after checkpoints and logs are synced.

## What must be uploaded

Upload the repository with these paths:

- `configs/`, `src/`, `scripts/`, `tests/`;
- `requirements-gpu.txt`, `DATASET_CARD.md`, `DATA_LICENSES.md`;
- `data/processed/manifests/`;
- `data/processed/audio_16k/`;
- `data/reports/` (bao gồm `audio_inventory.jsonl` để xác minh từng FLAC sau upload).

Do not upload `.venv/`, `data/processed/audio/` (unstandardized duplicates), `data/review/`, caches, or old model checkpoints. The clean manifests use project-relative POSIX paths, so the repository can be mounted anywhere on Linux.

## Environment

Use a current Linux PyTorch/CUDA image with Python 3.11. From the repository root:

```bash
python -m venv .venv-gpu
source .venv-gpu/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu.txt
python - <<'PY'
import torch
print({"torch": torch.__version__, "cuda": torch.cuda.is_available(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
assert torch.cuda.is_available()
PY
python scripts/preflight_project.py --gate gpu --output data/reports/preflight_gpu_cloud.json
```

GPU preflight strict-parse và stable-hash config/report/manifest/inventory, so sánh SHA-256 của sáu manifest với `artifact_lock.yaml`, sau đó băm hai lượt trên cùng regular no-link file cho toàn bộ 31.929 FLAC theo `audio_inventory.jsonl`. Inventory từ chối path escape, link/junction, ID/path/hash trùng, byte/digest sai và file đổi trong lúc đọc. Không bắt đầu billed dry-run nếu một file thiếu, sai size hoặc sai hash; dự kiến preflight lâu hơn một lượt hash thường.

Inventory phải được tái tạo bằng `scripts/build_audio_inventory.py`: producer strict-parse ba local manifest, khóa role/ID/canonical audio path, stable-hash file trước khi kiểm duplicate và chỉ publish inventory + summary crash-durable sau khi toàn bộ gate pass. Không chỉnh tay inventory hoặc cập nhật lock từ file sinh dở.

Record `nvidia-smi`, package versions and the preflight JSON with every run.

## AIMET handoff for QCS6490

Do not reuse the official OpenAI Whisper encodings for PhoWhisper or a
fine-tuned checkpoint. Encodings depend on the actual weights. On a Linux
machine, create a separate recipe environment and calibrate the encoder from
the audited train manifest:

```bash
python -m venv .venv-aihub-models
source .venv-aihub-models/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-aihub-models.txt
python scripts/prepare_phowhisper_aimet.py \
  --component encoder \
  --checkpoint models/asr/phowhisper-small-medical \
  --manifest data/processed/manifests/asr--train-local.jsonl \
  --limit 32 \
  --output-dir models/aihub/phowhisper-small-medical/encoder.aimet

python scripts/prepare_phowhisper_aimet.py \
  --component decoder \
  --checkpoint models/asr/phowhisper-small-medical \
  --manifest data/processed/manifests/asr--train-local.jsonl \
  --limit 32 \
  --decoder-steps-per-row 8 \
  --output-dir models/aihub/phowhisper-small-medical/decoder.aimet
```

The entrypoint checks that every calibration row is `train`, keeps paths inside
the workspace, selects rows deterministically, requires mono 16 kHz audio, and
sets the Qualcomm recipe checkpoint only in memory. It fails closed on Windows
before importing AIMET and does not edit `site-packages`.

Run the encoder and decoder as separate component jobs to keep peak memory
bounded. The decoder path runs the optimized float encoder, then advances the
optimized float decoder with the audited transcript tokens (teacher forcing).
It validates all 51 Qualcomm input names, shapes, and dtypes before AIMET sees a
batch. Raw audio, transcript text, token IDs, and KV-cache tensors stay in
memory and are never written to the output directory; only the AIMET checkpoint
and non-sensitive row-ID hash/count evidence are saved. Do not substitute
random caches or label an encoder-only artifact as a deployable ASR model.

The decoder command is locally unit-tested but cannot be executed on Windows.
On Linux, first use a small `--limit 2 --decoder-steps-per-row 2` smoke. Only
increase the sample count after both component checkpoints are created and the
evidence reports `sensitive_calibration_tensors_persisted: false`.

## GPU rounds under the resource controller

The authoritative entry point runs one model at a time, enforces the 35% per-process VRAM allocation and 40% hard VRAM stop, keeps rolling utilization at 70%, resumes at 55%, and reacts immediately at 74%. Every threshold remains strictly below 75%; there is no separate early-morning policy. Before an initial GPU training or evaluation child, the controller waits without using GPU until current utilization is at or below the active resume threshold and free VRAM can accommodate the full 40% hard process budget. A resumed round with at least 50 historical samples attributed by exact process-group PID or Linux `NSpid` instead reserves the largest observed process peak plus the greater of 25% or 2 GiB, still capped by the 40% hard budget; post-spawn growth telemetry is never trusted for this estimate. The gate requires two consecutive safe capacity samples, preventing a transient one-sample dip from launching a child into renewed contention. It appends one capacity-evidence record per minute and owns the stage PID while waiting, so a second controller cannot launch the same stage. The controller then snapshots NVIDIA compute-process memory immediately before spawning each trainer. It normally attributes VRAM by process-group PID; when a container exposes host PIDs through `nvidia-smi` but container PIDs through `ps`, it instead follows Linux `NSpid` aliases. Only if neither exact mapping exists does it retain post-spawn growth as telemetry, never as a hard-stop or capacity estimate. Every monitor record states the attribution mode and both tracked PID sets, so a zero measurement cannot silently hide a PID-namespace mismatch.

Interrupted training is retried at most three total attempts and only from a checkpoint whose watcher index, archive, checksum sidecar, member manifest, Trainer state, and live files all agree. Each attempt has a distinct log and the state preserves prior return codes and timestamps. If no verified checkpoint exists, or three attempts fail, the controller stops fail-closed instead of silently retraining from scratch.

```bash
python scripts/run_gpu_rounds.py --task mt --config configs/gpu_rounds.yaml --output-root gpu-runs
python scripts/run_gpu_rounds.py --task asr --config configs/gpu_rounds.yaml --output-root gpu-runs
```

The ASR command runs the three configured pilots, selects the lowest completed validation WER, inherits its learning rate and LoRA parameters, then appends `final-selected-full` automatically. That final candidate uses all training rows, 512 validation rows, three epochs and effective batch 32. It is a full-data LoRA fine-tune, not a full-parameter fine-tune. Every checkpoint and completed round is archived with SHA-256 evidence before the next model may start. The checkpoint watcher writes an atomic archive, checksum sidecar and canonical content manifest containing each member path, byte count and SHA-256, then reads the archive back against that manifest before publishing it in `checkpoint_archives.json`. A checksum-verified legacy checkpoint is backfilled only from its still-live checkpoint tree and must pass the same read-back verification.

When an MT full-run is already active, schedule the remaining program once instead of launching later stages by hand:

```bash
nohup ../.venv-onevoice/bin/python scripts/run_gpu_program.py \
  --state-dir gpu-runs/program-YYYYMMDD-HHMMSS \
  --mt-run gpu-runs/mt-YYYYMMDD-HHMMSS \
  --wait-pid MT_CONTROLLER_PID \
  > gpu-runs/program-YYYYMMDD-HHMMSS.launcher.log 2>&1 &
```

The orchestrator waits without using the GPU, verifies the completed MT summary and selected adapter, and examines only the latest validation-loss trend. If the last four validation points are monotonic, the newest is the best, and their relative gain is at least 0.3%, it warm-starts a continuation from the best adapter with a fresh optimizer/scheduler, half learning rate (`5e-5`), at most two additional epochs, and early stopping patience 4. This keeps the combined budget at three epochs without reloading the exhausted one-epoch LR scheduler. Otherwise it proceeds directly to the locked MT candidate suite. Locked test results never decide whether to train longer. It then runs ASR pilots → automatically selected full-data ASR → full locked ASR candidate suite. Candidate exit code 2 is recorded as a valid gate failure and does not prevent the independent ASR training from running; infrastructure/training errors stop the chain fail-closed. `program_state.json` is written atomically before and after every stage. Completion of this chain only records `candidate_a_gate_pass`; `promotion_allowed` remains false until the multi-model bake-off, one-time blind v2 test and QCS6490 deployment gate all pass.

If the lightweight orchestrator exits while its state is still `waiting_for_mt`, restart it with the same arguments plus `--resume`. Fail-closed recovery is also available when an out-of-band MT/ASR candidate or the ASR multi-round trainer has completed. Candidate recovery verifies `candidate_gate.json`, the archive, checksum sidecar, content manifest, adapter binding and every required local evidence file. Each prediction provenance record is revalidated internally and must bind its task, row count, bytes and SHA-256 to the exact current adapter-tree manifest, not merely the same adapter path. ASR-training recovery verifies every round archive/sidecar/manifest plus the complete selected-adapter directory against the archived path set, bytes and SHA-256 before continuing to the candidate gate; missing, unexpected, modified or symlinked adapter content is rejected. Gate, reports, predictions, provenance and resource monitors must match byte-for-byte. A wrapper may append its final terminal JSON to a `.log` after the immutable archive is closed; that local log divergence is recorded but the verified archived log remains authoritative. Missing, still-running, modified or incomplete non-log evidence is rejected before state mutation or GPU launch. Any state outside these explicit terminal-evidence paths remains non-resumable so a trainer/evaluator cannot be duplicated.

## Legacy manual 100-step billed dry-runs

The limits avoid preprocessing all data merely to test VRAM and throughput.

```bash
python -m src.training.finetune_whisper_vi \
  --precision fp16 --batch-size 4 --eval-batch-size 4 \
  --gradient-accumulation-steps 8 --max-steps 100 \
  --limit-train 1024 --limit-validation 256 \
  --output-dir models/dryrun/asr-phowhisper-small

python -m src.training.finetune_mt_medical \
  --direction joint --precision fp16 --batch-size 4 --eval-batch-size 4 \
  --gradient-accumulation-steps 8 --max-steps 100 \
  --limit-train 2048 --limit-validation 512 \
  --output-dir models/dryrun/mt-nllb-joint
```

For a 24 GB card, change both batch sizes to 2 and gradient accumulation to 16. If that still OOMs, try batch 1 with accumulation 32 before changing model or context length.

Capture elapsed seconds, peak VRAM (`nvidia-smi --query-compute-apps=used_memory --format=csv`), examples/second and loss trend. Extrapolate total cost from the measured steps/second, then decide whether to continue. Do not silently truncate data or sequence length.

## Legacy manual fine-tunes

ASR uses true on-the-fly speed/gain/noise augmentation only for train batches. Validation audio is untouched.

```bash
python -m src.training.finetune_whisper_vi \
  --precision fp16 --batch-size 4 --eval-batch-size 4 \
  --gradient-accumulation-steps 8 --epochs 5 \
  --output-dir models/asr/phowhisper-small-medical
```

MT creates one deployment-compatible joint EN↔VI checkpoint. Each MedEV pair is seen once in each direction; in-training checkpoint selection uses validation loss, while BLEU/chrF/safety generation is run separately per direction.

```bash
python -m src.training.finetune_mt_medical \
  --direction joint --precision fp16 --batch-size 4 --eval-batch-size 4 \
  --gradient-accumulation-steps 8 --epochs 1 \
  --output-dir models/mt/nllb-medical
```

To resume the newest checkpoint in the same output directory:

```bash
python -m src.training.finetune_whisper_vi --precision fp16 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 8 --epochs 5 --resume-from-checkpoint
python -m src.training.finetune_mt_medical --direction joint --precision fp16 --batch-size 4 --eval-batch-size 4 --gradient-accumulation-steps 8 --epochs 1 --resume-from-checkpoint
```

Do not run ASR and MT concurrently on one GPU. Sync each output directory and `trainer_state.json` to persistent/off-instance storage before starting the next job.

After both selected checkpoints are synced into the configured model paths, set
`runtime.allow_base_model_fallback: false` in `configs/pipeline_config.yaml`.
Then a missing/corrupt fine-tuned checkpoint aborts startup instead of silently
running the base model.

## Acceptance after fine-tune

1. Generate predictions for the complete locked validation split and the untouched locked test only after selecting a checkpoint.
2. Report ASR normalized WER plus source/accent/role/recording/code-switch slices and bootstrap confidence intervals.
3. Report MT SacreBLEU, chrF2, number/unit/negation preservation and both directions separately.
4. Compare base float → fine-tuned float → INT8 export. Reject quantization if medical entity/number safety regresses materially.
5. Run at least 100 human-reviewed medical translations and the TTS medication/number pronunciation suite before any demo claim.
6. Keep the safety guard fail-closed: unsafe output is displayed for confirmation and is not spoken automatically.

GPU execution has started under the controller above. Do not launch these legacy manual commands concurrently with the managed MT/ASR sequence; use them only for isolated diagnosis after confirming no controller-owned trainer is active.
