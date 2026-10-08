# OneVoice — MediVoice Edge (Healthcare Voice AI)

Prototype dịch giọng nói y tế Việt ↔ Anh hướng tới chạy on-device trên Qualcomm QCS6490. Đây là dự án do **một người** phát triển; dữ liệu đã vượt EDA/QA gate và GPU fine-tune theo phiên đang được điều phối, lưu checkpoint và kiểm chứng độc lập.

> Trạng thái kiểm tra lại 08/10/2026: **data/GPU-ready; Candidate A đã huấn luyện xong nhưng fail clinical gate; bake-off chưa hoàn tất; release gate vẫn khóa**. NLLB-600M và PhoWhisper-small Candidate A chỉ được giữ làm reference, không được promotion. M2M100 EN→VI zero/r8/r16/r32 và VI→EN zero/r8/r16 đều hoàn tất nhưng fail clinical gate. Pilot VI→EN r32 kết thúc bất thường gần bước 110/400; checkpoint 100 đã được tải và xác minh, nhưng chưa có checkpoint 200 và không được coi là kết quả benchmark. Recovery runner gần nhất dừng fail-closed trước training vì interpreter alias tương đối được kiểm tra theo sai working directory; lỗi đã được sửa local cùng đường `--resume-audit` chỉ đọc, còn remote chưa được resume. Whisper-small multilingual và PhoWhisper-base vẫn chờ chạy tuần tự; blind v2, INT8/QNN parity và latency toàn tuyến trên QCS6490 vật lý chưa hoàn tất.

## 📁 Cấu trúc Tài liệu Dự án

- `project_analysis.md` — nguồn sự thật về kiến trúc, roadmap, trạng thái và quyết định
- `configs/datasets.yaml` — registry dữ liệu, license, split và vai trò train/eval
- `scripts/audit_datasets.py` — EDA metadata đầy đủ trước merge
- `scripts/merge_manifests.py` — merge có khóa chống leakage
- `scripts/materialize_audio.py` — tải audio sau khi EDA được duyệt
- `scripts/qc_local_audio.py` — decode, chuẩn hóa FLAC mono 16 kHz và loại near-silence/CPS bất khả thi
- `scripts/run_baseline_benchmarks.py` — baseline ASR VI/EN và MT trên tập khóa
- `scripts/run_asr_candidate_suite.py` — chấm LoRA ASR Việt trên full test, code-switch và clinical audio suite, fail-closed
- `scripts/run_mt_candidate_suite.py` — chấm LoRA MT trên full locked test + clinical suite, fail-closed
- `scripts/verify_checkpoint_download.py` — xác minh lại archive checkpoint đã tải, sidecar, content manifest, checkpoint index và `trainer_state` mà không extract file
- Mỗi candidate suite xuất bundle bằng chứng gồm archive, SHA-256 sidecar và manifest từng file/bytes/checksum; bundle được đọc lại và xác minh trước khi công bố
- `scripts/run_gpu_program.py` — tự quyết định MT continuation bằng validation trend, rồi nối tuần tự MT gate → ASR pilots/full-data → ASR gate; state nguyên tử và không chạy hai GPU child đồng thời
- `scripts/run_model_bakeoff.py` — sau khi Candidate A hoàn tất, đóng băng nó rồi chạy successive-halving đa model trên selection dev; không mở test khóa cũ để chọn model
- `scripts/run_blind_candidate_suite.py` — khóa checksum và mở blind test v2 đúng một lần cho winner cuối
- `scripts/prepare_deployment_benchmark.py` — tạo draft đã khóa winner/adapter và chỉ finalize evidence QCS6490 sau khi measurement artifact bất biến khớp board/model, tự tính latency/power/thermal và vượt physical-board gate
- `scripts/capture_qcs6490_identity.py` — chạy trên board thật để khóa device-tree/sysfs ARM64; Arduino/cloud metadata không được tính là evidence QCS6490
- `scripts/capture_qcs6490_runtime.py` — chạy command inference trực tiếp trên QCS6490, lấy latency cùng power/thermal sysfs và peak RSS thành raw evidence bất biến; không đi qua shell
- `scripts/capture_compiled_predictions.py` — chạy compiled inference bằng argv không qua shell và khóa artifact/manifest/decoding/prediction checksum vào provenance bất biến trước parity gate
- `scripts/seal_quantization_parity.py` — tái chấm float/compiled predictions trên cùng manifest khóa, paired-bootstrap metric và zero-regression bảy slice lâm sàng trước khi khóa parity evidence vào deployment gate
- `scripts/seal_qcs6490_measurement.py` — chạy trên board thật để kiểm live identity, raw latency/power/thermal, timestamp, memory và compiled artifact trước khi phát hành measurement evidence bất biến
- `scripts/create_verified_backup.py` — tạo backup data/model/report nguyên tử; manifest v2 khóa SHA-256 từng file/archive và terminal `comparison.json`, cấm symlink/special member rồi tự đọc lại toàn bộ tar trước khi công bố; bake-off không chuyển sang terminal nếu backup chưa verify và commit local/tracking/live remote chưa trùng nhau
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

## Kế hoạch thực thi tiếp theo

1. Trước recovery, chủ động đưa remote tracked checkout tới một HEAD đã review có bản sửa interpreter alias và `--resume-audit`; chạy audit dưới đúng interpreter. Chỉ khi audit pass và không có GPU child sống mới khởi động đúng một runner.
2. Resume M2M100 VI→EN r32 chỉ từ checkpoint 100 đã xác minh; tải và kiểm tra archive/manifest/sidecar/index ngay khi checkpoint 200/300/400 xuất hiện. Không biến lần dừng bất thường cũ thành benchmark evidence.
3. Tiếp tục ASR challenger theo đúng thứ tự Whisper-small multilingual rồi PhoWhisper-base. VinAI AGPL vẫn khóa khi chưa có phê duyệt research-license rõ ràng và không đủ điều kiện production.
4. Chỉ chọn winner sau hard safety gate và luật đa metric/95% CI; blind v2 chỉ được mở một lần trên dữ liệu chưa từng thấy và phải fail-closed khi dữ liệu chưa sẵn sàng.
5. Sau khi có winner, chạy INT8 parity/QNN và đo latency, memory, power, thermal trên QCS6490 vật lý. Arduino, cloud host hoặc profile model tham chiếu không thay thế được evidence board thật.
6. Giữ Piper CPU/text-only fallback an toàn trong khi hoàn thiện exporter; mọi fallback phải giữ ASR/MT safety bắt buộc và không tự phát audio lâm sàng khi chưa có xác nhận.
