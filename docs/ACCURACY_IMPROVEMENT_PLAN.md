# Kế hoạch nâng accuracy và an toàn lâm sàng

Ngày khóa kế hoạch: 27/09/2026.

## Kết luận điều hành

Accuracy được tối ưu theo hai trục độc lập: chất lượng tổng thể (WER/CER, BLEU/chrF) và bảo toàn thông tin lâm sàng. Checkpoint chỉ được phát hành khi cả hai trục đều đạt gate. Fine-tune có thể cải thiện thuật ngữ y tế, code-switch và câu chuyên ngành, nhưng không được xem là cơ chế an toàn; tên thuốc, liều, số, đơn vị và phủ định luôn có test khóa riêng.

Baseline đang dùng làm mốc: ASR Việt WER 20,87%, ASR Anh WER 25,54%, MT BLEU 23,76 và chrF2 44,96. Mẫu ASR Việt code-switch có WER 23,66%. Đây là các số đo trên 24 mẫu mỗi benchmark, nên vòng đánh giá GPU phải tăng kích thước mẫu và báo confidence interval trước khi tuyên bố cải thiện.

## Thứ tự thực hiện mạnh nhất

1. Khóa dữ liệu và leakage: giữ test/clinical suite ngoài train, checksum manifest, speaker/recording/audio-hash split cho ASR và exact-pair split cho MT.
2. Nâng dữ liệu train: ưu tiên có kiểm soát các hàng chứa thuốc, liều, số, đơn vị, phủ định, thuật ngữ chuyên khoa và code-switch; chỉ oversample train với hệ số 2, không nhân validation/test.
3. ASR Việt: chạy ba pilot LoRA để chọn learning rate/rank theo WER validation, sau đó tự động chạy LoRA full-data ba epoch từ cấu hình tốt nhất; chỉ thử full-parameter fine-tune như một candidate riêng nếu tài nguyên và locked evaluation chứng minh có lợi. Chọn checkpoint cuối theo WER validation và các slice code-switch/clinical token preservation, không theo test.
4. ASR Anh: không train trên Eka test. Chỉ fine-tune khi có corpus train y tế tiếng Anh có license và speaker split; trước đó dùng distil-small.en và theo dõi WER riêng.
5. MT: LoRA joint EN↔VI để tìm cấu hình, sau đó chạy candidate full-data NLLB trên MedEV train; giữ greedy nếu validation tiếp tục tốt hơn beam. Không dùng test để chọn checkpoint và không gọi LoRA full-data là full-parameter fine-tune.
6. Chạy suite lâm sàng khóa hai chiều. Bất kỳ lỗi thuốc/liều/số/đơn vị/phủ định/thuật ngữ nào đều chặn release và được định tuyến fail-closed.
7. Chạy parity sau quantization, sau đó benchmark QCS6490 thật. Không suy ra accuracy/latency on-device từ CPU hoặc cloud profile.

## Gate phát hành

Các ngưỡng máy đọc được nằm ở `configs/accuracy_program.yaml`. Mục tiêu vòng kế tiếp là ASR Việt WER ≤ 19%, ASR Anh WER ≤ 23%, MT BLEU ≥ 25, chrF2 ≥ 46, code-switch WER ≤ 21%, và 0 lỗi phát hiện được trên mọi slice safety. Ngưỡng 0 lỗi không chứng minh hệ thống an toàn; nó chỉ là điều kiện tối thiểu để tiếp tục human review.

## Lệnh local đã chuẩn hóa

```powershell
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m compileall -q src scripts demo tests

# Fine-tune thật nhưng chỉ là CPU smoke, không được promote
.venv\Scripts\python.exe -m src.training.finetune_whisper_vi --method lora --cpu-smoke --local-files-only --precision fp32 --max-steps 1 --limit-train 2 --limit-validation 1 --batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 1 --clinical-oversample-factor 2 --output-dir models/asr/phowhisper-small-medical-cpu-smoke
.venv\Scripts\python.exe -m src.training.finetune_mt_medical --method lora --cpu-smoke --local-files-only --precision fp32 --max-steps 1 --limit-train 2 --limit-validation 1 --batch-size 1 --eval-batch-size 1 --gradient-accumulation-steps 1 --clinical-oversample-factor 2 --output-dir models/mt/nllb-medical-cpu-smoke

# Benchmark safety và gate tổng hợp
.venv\Scripts\python.exe scripts/run_baseline_benchmarks.py --task mt --manifest data/eval/medical_safety_mt.jsonl --samples 32 --batch-size 2 --num-beams 1 --name mt_clinical_base --output-dir data/reports/accuracy
.venv\Scripts\python.exe scripts/run_accuracy_gates.py

# Sau full-run: chấm adapter trên toàn bộ test khóa + toàn bộ suite lâm sàng,
# điều tiết GPU theo cùng lịch 38%/75%, rồi đóng gói report có SHA-256.
.venv\Scripts\python.exe scripts/run_asr_candidate_suite.py --adapter gpu-runs/asr-YYYYMMDD-HHMMSS/NN-final-selected/model --device cuda --precision bf16 --batch-size 4 --num-beams 1
.venv\Scripts\python.exe scripts/run_mt_candidate_suite.py --adapter gpu-runs/mt-YYYYMMDD-HHMMSS/01-final-r8-lr1e4-full/model --device cuda --precision bf16 --batch-size 8 --num-beams 4
```

CPU smoke ghi `promotion_allowed=false` trong `training_run.json`. Chỉ checkpoint full-run trên GPU, vượt locked validation/test và suite lâm sàng mới được thay đường dẫn production.

`--samples 0` trong `run_baseline_benchmarks.py` có nghĩa là chấm toàn bộ manifest. Khi có `--adapter`, benchmark nạp PEFT/LoRA lên model nền đã pin revision; mặc định cũ vẫn là CPU/fp32. `run_asr_candidate_suite.py` xác minh cả full ASR test và `medical_safety_asr_vi.jsonl`, đồng thời chứng minh 16 audio safety là bản sao nguyên trạng của locked test trước khi suy luận. Bộ này chấm bảo toàn cụm từ đã duyệt cho tên thuốc, liều, số, đơn vị, phủ định, thuật ngữ và code-switch; full WER, code-switch WER hoặc một slice bất kỳ không đạt đều khóa promotion. `run_mt_candidate_suite.py` áp dụng cùng nguyên tắc cho MT. Hai script lưu predictions/report/log/resource monitor, rồi tạo `tar.gz` và sidecar `.sha256`. Kết quả fine-tune tốt nhưng gate lâm sàng fail vẫn không được promotion; 0 lỗi phát hiện được cũng không phải chứng minh an toàn.

## Điều phối GPU theo giờ

`scripts/run_gpu_rounds.py` chạy tuần tự các vòng LoRA trong `configs/gpu_rounds.yaml`, giới hạn process ở 35% VRAM và điều tiết bằng `SIGSTOP/SIGCONT`. Ngoài 02:00–09:00 Asia/Bangkok, rolling GPU utilization bị giữ dưới 38%. Trong 02:00–09:00, ngưỡng được nâng tự động lên 75%; giới hạn VRAM không đổi. Pilot chỉ tokenize tập ưu tiên lâm sàng đủ lớn cho số bước cấu hình (MT 8.192, ASR 6.144) để tránh lãng phí CPU. ASR pilot dùng 512 validation rows gồm cả VietMed và ViMedCSS/code-switch thay vì chọn cấu hình trên mẫu nhỏ 64 dòng. Controller chọn pilot có `eval_wer` thấp nhất, kế thừa learning rate/rank/alpha/dropout rồi tự nối vòng `final-selected-full`: toàn bộ train, 512 validation, ba epoch, effective batch 32. MT full-data dùng cấu hình đã khóa riêng. Mỗi vòng sinh log tài nguyên, `training_run.json`, tar.gz và SHA-256 để tải về ngay trước vòng kế tiếp. `scripts/watch_checkpoints.py` chạy cạnh từng round, chỉ nhận checkpoint khi `trainer_state.global_step` khớp tên thư mục và đủ model/optimizer/scheduler/RNG, sau đó tạo tar.gz nguyên tử, đọc lại cấu trúc và sinh sidecar SHA-256. Hậu kiểm candidate dùng lại chính bộ điều tiết GPU này, nên việc tăng công suất ban đêm không bỏ qua giới hạn tài nguyên.

`scripts/run_gpu_program.py` là state machine cho phần còn lại của chương trình dài ngày: chờ MT controller hiện tại kết thúc, xác minh selected adapter, rồi quyết định có continuation MT hay không chỉ từ bốn điểm validation loss cuối. Hệ thống chỉ kéo dài khi loss giảm đơn điệu, điểm mới nhất tốt nhất và relative gain đạt ít nhất 0,3%. Continuation warm-start từ best adapter với optimizer/scheduler mới, learning rate giảm còn `5e-5`, tối đa hai epoch bổ sung và early stopping patience 4; cách này giữ tổng ngân sách tối đa ba epoch mà không nạp lại scheduler một epoch đã cạn. Locked test tuyệt đối không được dùng để quyết định train thêm. Sau đó state machine chạy locked MT candidate suite, ASR pilots/full-data và locked ASR suite. Các GPU child luôn tuần tự; gate fail được lưu như bằng chứng và không bị nhầm với lỗi hạ tầng. State ghi nguyên tử cho từng stage, còn `promotion_allowed` chỉ đúng khi cả hai candidate đều vượt gate. Việc tự nối stage không thay thế tải archive/checksum về local ngay khi chúng xuất hiện.

## Backup và phục hồi

`scripts/create_verified_backup.py` tạo ba tar độc lập cho data (không lặp reports), models và reports, sinh `SHA256SUMS.txt`, rồi đọc lại toàn bộ archive để xác minh checksum và số file. Backup nằm trong `.backups/`, không push lên Git vì chứa dữ liệu/model lớn và có ràng buộc license.
