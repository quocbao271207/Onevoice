# OneVoice — MediVoice Edge (Healthcare Voice AI)

Prototype dịch giọng nói y tế Việt ↔ Anh hướng tới chạy on-device trên Qualcomm QCS6490. Đây là dự án do **một người** phát triển; dữ liệu đã vượt EDA/QA gate và GPU fine-tune theo phiên đang được điều phối, lưu checkpoint và kiểm chứng độc lập.

> Trạng thái kiểm tra lại 09/10/2026: **data/GPU-ready; Candidate A đã huấn luyện xong nhưng fail clinical gate; bake-off chưa hoàn tất; release gate vẫn khóa**. NLLB-600M và PhoWhisper-small Candidate A chỉ được giữ làm reference, không được promotion. M2M100 EN→VI zero/r8/r16/r32 và VI→EN zero/r8/r16 đều hoàn tất nhưng fail clinical gate. Pilot VI→EN r32 kết thúc bất thường gần bước 110/400; checkpoint 100 đã được tải và xác minh, nhưng chưa có checkpoint 200 và không được coi là kết quả benchmark. Waiter 375401 đã gặp capacity an toàn rồi `exec` runner, nhưng runner remote cũ dừng fail-closed tại completed EN→VI r8 do interpreter alias bị coi là command change, trước khi chạm VI→EN r32. Bản sửa strict training-alias và `--resume-audit` đã ở local/origin mới hơn; remote chưa được cập nhật hoặc resume lại. Whisper-small multilingual và PhoWhisper-base vẫn chờ chạy tuần tự; blind v2, INT8/QNN parity và latency toàn tuyến trên QCS6490 vật lý chưa hoàn tất.

## 📁 Cấu trúc Tài liệu Dự án

- `project_analysis.md` — nguồn sự thật về kiến trúc, roadmap, trạng thái và quyết định
- `configs/datasets.yaml` — registry dữ liệu, license, split và vai trò train/eval
- `scripts/audit_datasets.py` — EDA metadata đầy đủ trước merge; URL tải ký chỉ tồn tại trong RAM và không được ghi vào record/listening report
- `scripts/merge_manifests.py` — merge có khóa chống leakage
- `scripts/materialize_audio.py` — refresh URL từ đúng Hugging Face datasets-server ngay trước khi tải, chỉ nhận public HTTPS và loại URL khỏi local manifest
- `scripts/sanitize_asr_report_urls.py` — dry-run mặc định để kiểm artifact EDA cũ; chỉ `--apply` mới loại nguyên tử đúng trường `audio_url`
- `scripts/qc_local_audio.py` — decode, chuẩn hóa FLAC mono 16 kHz và loại near-silence/CPS bất khả thi
- `scripts/run_baseline_benchmarks.py` — baseline/candidate ASR VI/EN và MT trên manifest strict/bounded đã khóa; prediction, provenance và report ghi crash-durable, resume-scoring chỉ dùng evidence exact-schema/path/bytes/SHA-256/rows đã đọc ổn định
- `scripts/run_asr_candidate_suite.py` — chấm LoRA ASR Việt trên full test, code-switch và clinical audio suite, fail-closed
- `scripts/run_mt_candidate_suite.py` — chấm LoRA MT trên full locked test + clinical suite, fail-closed
- `scripts/verify_checkpoint_download.py` — xác minh lại archive checkpoint đã tải, sidecar, content manifest, checkpoint index và `trainer_state` mà không extract file; mọi input strict/bounded/no-link, identity được khóa ổn định trước/sau tar read và receipt chỉ publish crash-durable hoặc reuse khi bằng chứng khớp exact
- Mỗi candidate suite xuất bundle bằng chứng gồm archive, SHA-256 sidecar và manifest từng file/bytes/checksum; writer dùng temp ngẫu nhiên đã fsync/verify rồi hard-link publish độc quyền, retry chỉ reuse bundle khớp live tree hoặc phục hồi sidecar thiếu mà không thay archive. Verifier từ chối link/junction, áp byte/member/content cap, strict-parse manifest, stable-hash toàn bundle và đọc/hash lại archive + sidecar + manifest sau khi duyệt tar trước khi công bố
- `scripts/watch_checkpoints.py` — chỉ nhận completed Trainer checkpoint có strict stable `trainer_state`, archive qua cùng transaction độc quyền với candidate evidence, rồi cập nhật checkpoint index crash-durable; index hỏng/duplicate không bị âm thầm ghi đè
- `scripts/run_gpu_program.py` — tự quyết định MT continuation bằng validation trend, rồi nối tuần tự MT gate → ASR pilots/full-data → ASR gate; state strict/bounded được ghi crash-durable, resume chỉ nhận summary/trainer/gate/provenance/prediction/archive và toàn bộ adapter tree đã xác minh ổn định, đồng thời không chạy hai GPU child
- `scripts/run_model_bakeoff.py` — sau khi Candidate A hoàn tất, đóng băng nó rồi chạy successive-halving đa model trên selection dev; không mở test khóa cũ để chọn model; config/state/gate/evidence control-plane được đọc bounded, strict và chống link/TOCTOU trước resume, còn state transition được ghi atomically với temp độc quyền, fsync và verify lại
- `scripts/run_blind_candidate_suite.py` — khóa checksum và mở blind test v2 đúng một lần cho winner cuối; transaction theo slot có mutex chống mở trùng, mọi lock/manifest/selection/report/provenance được parse và hash từ cùng payload ổn định strict/bounded, adapter winner được quét bounded không link/special-file và streaming-hash hai tree pass, audio ASR được tái hash bounded mỗi lần verify/resume, còn state/gate/provenance được ghi crash-durable qua writer dùng chung
- `scripts/prepare_deployment_benchmark.py` — tạo draft đã khóa winner/adapter cùng exact stable payload/path/SHA-256 của selection comparison bằng writer độc quyền crash-durable; final report bắt buộc đọc lại selection binding đó, rồi chỉ finalize evidence QCS6490 sau khi strict JSON, stable SHA-256/size của identity, parity, measurement và compiled artifact khớp board/model, tự tính latency/power/thermal và vượt physical-board gate
- `scripts/capture_qcs6490_identity.py` — chạy trên board thật để khóa device-tree/sysfs ARM64 vào evidence exclusive crash-durable; Arduino/cloud metadata không được tính là evidence QCS6490
- `scripts/capture_qcs6490_runtime.py` — chạy command inference trực tiếp trên QCS6490, lấy latency cùng power/thermal sysfs và peak RSS thành raw evidence exclusive crash-durable; không đi qua shell
- `scripts/capture_compiled_predictions.py` — chạy compiled inference bằng argv không qua shell; stable-hash artifact/manifest trước và sau command, strict-parse predictions rồi publish provenance exclusive crash-durable trước parity gate
- `scripts/seal_quantization_parity.py` — strict-parse và stable-hash manifest/provenance/float/compiled predictions, tái chấm paired-bootstrap metric cùng zero-regression bảy slice lâm sàng, rồi publish parity evidence exclusive crash-durable vào deployment gate
- `scripts/seal_qcs6490_measurement.py` — chạy trên board thật để strict-parse và stable-hash live identity, raw latency/power/thermal, timestamp, memory và compiled artifact trước khi phát hành measurement evidence bất biến
- `scripts/create_verified_backup.py` — tạo backup data/model/report nguyên tử; manifest v2 exact-schema/bounded khóa SHA-256 từng file/archive và terminal `comparison.json`, tar byte-reproducible không mang UID/GID/user/mtime máy tạo, cấm symlink/special member rồi tự đọc lại toàn bộ trước khi công bố; source và tar đều được quét streaming để chặn credential/private key và URL ký trong report/log mà không in secret; bake-off không chuyển terminal nếu backup chưa verify, secret scan chưa pass hoặc commit local/tracking/live remote chưa trùng
- `demo/demo_web.py` — UI web chỉ bind `127.0.0.1`, có Host/CSRF/CSP/no-store, request/audio bounded, không lộ candidate fail safety và không tự phát audio; playback dùng token one-time 120 giây và confirmation server-side
- `src/utils/bounded_file.py` — boundary file dùng chung: chỉ nhận regular file có byte cap, không symlink/junction/reparse point; hỗ trợ đọc nhỏ hoặc SHA-256 streaming hai lượt và phát hiện file bị thay giữa lúc đọc/hash
- `src/pipeline/adapter_evidence.py` — khóa một canonical adapter identity dùng chung cho Candidate A, bake-off và blind: chống link/junction/special file ở mọi cấp, cap directory/entry/file/bytes, ba lượt enumerate cùng hai lượt stable hash để bắt tree/content mutation
- `src/pipeline/durable_file.py` — ghi byte evidence crash-durable qua temp sở hữu riêng, `fsync` và persisted revalidation; hỗ trợ atomic replace cho checkpoint có thể tái tạo hoặc hard-link độc quyền không overwrite cho artifact bất biến
- `src/pipeline/stable_json.py` — boundary JSON mapping dùng chung: strict UTF-8, cấm duplicate key/non-finite, stable bounded read, no-link path và optional SHA-256 binding
- `src/pipeline/stable_jsonl.py` — boundary JSONL evidence dùng chung: streaming strict parser có byte/line/row cap, reject duplicate/non-finite/link, bind rows với stable SHA-256/bytes và có thể bắt buộc khớp digest ngoài
- `src/pipeline/flash_cache.py` — cache cụm cấp cứu exact-only; custom JSON bị giới hạn byte/entry, strict-schema, reject duplicate key/non-finite/link và chỉ publish nguyên tử sau safety validation
- `src/training/` — preflight và fine-tune scripts; chỉ chạy training khi có GPU
- `GPU_HANDOFF.md` — lệnh dry-run/full-run/resume, budget và shutdown checklist cho GPU thuê
- `AIHUB_DEPLOYMENT_PLAN.md` — kế hoạch QCS6490: smoke, export, INT8, profile, gate trước GPU và benchmark board
- `docs/ACCURACY_IMPROVEMENT_PLAN.md` — kế hoạch accuracy, fine-tune hai giai đoạn và gate lâm sàng
- `docs/MODEL_BAKEOFF_RESULTS.md` — trạng thái, license gate, fairness, winner rule và kết quả đa model
- `configs/accuracy_program.yaml` — ngưỡng release máy đọc được
- `data/eval/medical_safety_asr_vi.jsonl` — 16 audio test khóa cho thuốc/liều/số/đơn vị/phủ định/thuật ngữ/code-switch
- `data/eval/medical_safety_mt.jsonl` — tập test khóa riêng cho thuốc/liều/số/đơn vị/phủ định
- `docs/proposals/` — bản Markdown Phase 2 lịch sử; hai DOCX gốc vẫn nằm nguyên vẹn dưới `docs/`

Quy trình dữ liệu bắt buộc: `audit → nghe mẫu/duyệt license → merge manifest → tải audio → signal QC → locked baseline → GPU preflight → thuê GPU`. Không dùng Eka `test` để train; không coi chat tiếng Việt đơn ngữ là cặp dịch. MedEV được phép dùng cho nghiên cứu/phi thương mại, không tái phân phối dữ liệu.

## Kiểm tra nhanh trên máy local

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src scripts demo tests
.venv\Scripts\python.exe scripts\preflight_project.py --gate gpu --output data\reports\preflight_gpu_ready.json
.venv\Scripts\python.exe scripts\run_accuracy_gates.py
```

Kết quả regression gần nhất được ghi trong `project_analysis.md`; không dùng con số test cũ trong README làm release gate. Ba baseline CPU và full base cascade đã pass; CLI text mode pass; GPU gate `ready=true` và xác minh đủ 31.929 FLAC (3.541.434.742 bytes). Ablation decoding local giữ greedy cho ASR/MT: beam search chậm hơn 3–4,8 lần nhưng không cải thiện validation đáng tin cậy; prompt y khoa tĩnh bị loại vì tạo một lỗi thuật ngữ nguy hiểm.

Demo web cục bộ chỉ được mở sau khi model artifact đã sẵn sàng; server không nhận bind address bên ngoài loopback:

```powershell
.venv\Scripts\python.exe demo\demo_web.py --port 8765
```

Mở `http://127.0.0.1:8765`. Audio không tự phát; nút playback chỉ xuất hiện sau safety gate và câu hành động lâm sàng yêu cầu xác nhận riêng.

## Kế hoạch thực thi tiếp theo

1. Trước recovery, chủ động đưa remote tracked checkout tới một HEAD đã review có bản sửa interpreter alias và `--resume-audit`; chạy audit dưới đúng interpreter. Chỉ khi audit pass và không có GPU child sống mới khởi động đúng một runner.
2. Resume M2M100 VI→EN r32 chỉ từ checkpoint 100 đã xác minh; tải và kiểm tra archive/manifest/sidecar/index ngay khi checkpoint 200/300/400 xuất hiện. Không biến lần dừng bất thường cũ thành benchmark evidence.
3. Tiếp tục ASR challenger theo đúng thứ tự Whisper-small multilingual rồi PhoWhisper-base. VinAI AGPL vẫn khóa khi chưa có phê duyệt research-license rõ ràng và không đủ điều kiện production.
4. Chỉ chọn winner sau hard safety gate và luật đa metric/95% CI; blind v2 chỉ được mở một lần trên dữ liệu chưa từng thấy và phải fail-closed khi dữ liệu chưa sẵn sàng.
5. Sau khi có winner, chạy INT8 parity/QNN và đo latency, memory, power, thermal trên QCS6490 vật lý. Arduino, cloud host hoặc profile model tham chiếu không thay thế được evidence board thật.
6. Giữ Piper CPU/text-only fallback an toàn trong khi hoàn thiện exporter; mọi fallback phải giữ ASR/MT safety bắt buộc và không tự phát audio lâm sàng khi chưa có xác nhận.
