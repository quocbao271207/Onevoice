# Kế hoạch nâng accuracy và an toàn lâm sàng

Ngày khóa kế hoạch: 27/09/2026.

## Kết luận điều hành

Accuracy được tối ưu theo hai trục độc lập: chất lượng tổng thể (WER/CER, BLEU/chrF) và bảo toàn thông tin lâm sàng. Checkpoint chỉ được phát hành khi cả hai trục đều đạt gate. Fine-tune có thể cải thiện thuật ngữ y tế, code-switch và câu chuyên ngành, nhưng không được xem là cơ chế an toàn; tên thuốc, liều, số, đơn vị và phủ định luôn có test khóa riêng.

Baseline đang dùng làm mốc: ASR Việt WER 20,87%, ASR Anh WER 25,54%, MT BLEU 23,76 và chrF2 44,96. Mẫu ASR Việt code-switch có WER 23,66%. Đây là các số đo trên 24 mẫu mỗi benchmark, nên vòng đánh giá GPU phải tăng kích thước mẫu và báo confidence interval trước khi tuyên bố cải thiện.

## Thứ tự thực hiện mạnh nhất

1. Khóa dữ liệu và leakage: giữ test/clinical suite ngoài train, checksum manifest, speaker/recording/audio-hash split cho ASR và exact-pair split cho MT.
2. Nâng dữ liệu train: ưu tiên có kiểm soát các hàng chứa thuốc, liều, số, đơn vị, phủ định, thuật ngữ chuyên khoa và code-switch; chỉ oversample train với hệ số 2, không nhân validation/test.
3. ASR Việt: chạy LoRA để tìm learning rate/batch/augmentation rẻ hơn, sau đó full fine-tune PhoWhisper-small từ cấu hình tốt nhất; chọn checkpoint theo WER validation và theo slice code-switch/clinical token preservation.
4. ASR Anh: không train trên Eka test. Chỉ fine-tune khi có corpus train y tế tiếng Anh có license và speaker split; trước đó dùng distil-small.en và theo dõi WER riêng.
5. MT: LoRA joint EN↔VI để tìm cấu hình, sau đó full fine-tune NLLB trên MedEV train; giữ greedy nếu validation tiếp tục tốt hơn beam. Không dùng test để chọn checkpoint.
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
```

CPU smoke ghi `promotion_allowed=false` trong `training_run.json`. Chỉ checkpoint full-run trên GPU, vượt locked validation/test và suite lâm sàng mới được thay đường dẫn production.

## Điều phối GPU theo giờ

`scripts/run_gpu_rounds.py` chạy tuần tự các vòng LoRA trong `configs/gpu_rounds.yaml`, giới hạn process ở 35% VRAM và điều tiết bằng `SIGSTOP/SIGCONT`. Ngoài 02:00–09:00 Asia/Bangkok, rolling GPU utilization bị giữ dưới 38%. Trong 02:00–09:00, ngưỡng được nâng tự động lên 75%; giới hạn VRAM không đổi. Mỗi vòng sinh log tài nguyên, `training_run.json`, tar.gz và SHA-256 để tải về ngay trước vòng kế tiếp.

## Backup và phục hồi

`scripts/create_verified_backup.py` tạo ba tar độc lập cho data (không lặp reports), models và reports, sinh `SHA256SUMS.txt`, rồi đọc lại toàn bộ archive để xác minh checksum và số file. Backup nằm trong `.backups/`, không push lên Git vì chứa dữ liệu/model lớn và có ràng buộc license.
