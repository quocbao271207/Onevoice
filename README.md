# OneVoice — MediVoice Edge (Healthcare Voice AI)

Prototype dịch giọng nói y tế Việt ↔ Anh hướng tới chạy on-device trên Qualcomm QCS6490. Đây là dự án do **một người** phát triển; dữ liệu đã vượt EDA/QA gate và GPU fine-tune theo phiên đang được điều phối, lưu checkpoint và kiểm chứng độc lập.

> Trạng thái kiểm tra lại 27/09/2026: **data/GPU-ready; local LoRA training path đã smoke pass; release gate đang khóa**. EDA, merge chống leakage, materialize, signal QC, human QA 48/48 và GPU preflight đều pass. Hai fine-tune CPU smoke một bước đã tạo adapter thật nhưng đều ghi `promotion_allowed=false`. Baseline chưa đạt WER/BLEU target; suite lâm sàng 32 chiều còn fail phủ định, thuật ngữ và code-switch dù giữ đúng thuốc/liều/số/đơn vị trong mẫu này. Whisper Small w8a16 tham chiếu đã profile thành công trên NPU; PhoWhisper, Piper và NLLB vẫn còn các BYOM/QNN blocker nêu trong `project_analysis.md`. Latency toàn tuyến trên board vật lý chưa đo.

## 📁 Cấu trúc Tài liệu Dự án

- `project_analysis.md` — nguồn sự thật về kiến trúc, roadmap, trạng thái và quyết định
- `configs/datasets.yaml` — registry dữ liệu, license, split và vai trò train/eval
- `scripts/audit_datasets.py` — EDA metadata đầy đủ trước merge
- `scripts/merge_manifests.py` — merge có khóa chống leakage
- `scripts/materialize_audio.py` — tải audio sau khi EDA được duyệt
- `scripts/qc_local_audio.py` — decode, chuẩn hóa FLAC mono 16 kHz và loại near-silence/CPS bất khả thi
- `scripts/run_baseline_benchmarks.py` — baseline ASR VI/EN và MT trên tập khóa
- `scripts/run_asr_candidate_suite.py` — chấm LoRA ASR Việt trên full test, code-switch và clinical audio suite, fail-closed
- `scripts/run_mt_candidate_suite.py` — chấm LoRA MT trên full locked test + clinical suite, fail-closed và tạo archive checksum
- `scripts/run_gpu_program.py` — tự quyết định MT continuation bằng validation trend, rồi nối tuần tự MT gate → ASR pilots/full-data → ASR gate; state nguyên tử và không chạy hai GPU child đồng thời
- `src/training/` — preflight và fine-tune scripts; chỉ chạy training khi có GPU
- `GPU_HANDOFF.md` — lệnh dry-run/full-run/resume, budget và shutdown checklist cho GPU thuê
- `AIHUB_DEPLOYMENT_PLAN.md` — kế hoạch QCS6490: smoke, export, INT8, profile, gate trước GPU và benchmark board
- `docs/ACCURACY_IMPROVEMENT_PLAN.md` — kế hoạch accuracy, fine-tune hai giai đoạn và gate lâm sàng
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

1. Chạy specialized AIMET recipe trên Linux cho PhoWhisper weights; encoder handoff đã có, decoder còn phải tạo teacher-forced token/KV-cache calibration thật. Không retry generic cloud-PTQ graph.
2. Giữ Piper CPU fallback cho đến khi có source exporter loại được `NonZero/ReduceMax→Range`; graph mới phải qua local QNN gate trước khi upload. Sau đó mới export NLLB base theo encoder/decoder/KV-cache tĩnh.
3. Hoàn tất candidate MT full-data đang chạy, tải ngay checkpoint/report/checksum về local và chấm locked MT suite.
4. Chạy ba pilot ASR rồi để controller tự chọn cấu hình cho vòng ASR full-data; không chạy ASR và MT đồng thời trên một GPU.
5. Chỉ promotion candidate vượt cả metric tổng thể và locked clinical suite; sau đó chạy INT8 parity/QNN compile rồi mới benchmark toàn tuyến trên board QCS6490 thật.
