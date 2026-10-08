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

Challenge set thuật ngữ riêng gồm 128 lỗi validation đã xác minh: 110 lỗi MT có safety issue máy phát hiện và 18 lỗi ASR có word error thật, cân bằng hai chiều MT. Artifact `data/eval/terminology_challenge_set.jsonl` chỉ dùng để đánh giá; toàn bộ hàng mới luôn là `review_status=pending`, `evaluation_only=true` và `auto_correction_eligible=false`. `data/eval/terminology_challenge_set.report.json` khóa manifest, prediction source, archive/member provenance, số hàng và SHA-256. Human review có thể xác nhận hoặc loại từng candidate về sau, nhưng không được biến alias/hypothesis chưa duyệt thành post-correction.

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
.venv\Scripts\python.exe scripts/build_terminology_challenge_set.py

# Sau full-run: chấm adapter trên toàn bộ test khóa + toàn bộ suite lâm sàng,
# điều tiết GPU theo policy thống nhất rolling 70% / hard 74%, rồi đóng gói report có SHA-256.
.venv\Scripts\python.exe scripts/run_asr_candidate_suite.py --adapter gpu-runs/asr-YYYYMMDD-HHMMSS/NN-final-selected/model --device cuda --precision bf16 --batch-size 4 --num-beams 1
.venv\Scripts\python.exe scripts/run_mt_candidate_suite.py --adapter gpu-runs/mt-YYYYMMDD-HHMMSS/01-final-r8-lr1e4-full/model --device cuda --precision bf16 --batch-size 8 --num-beams 1
```

CPU smoke ghi `promotion_allowed=false` trong `training_run.json`. Chỉ checkpoint full-run trên GPU, vượt locked validation/test và suite lâm sàng mới được thay đường dẫn production.

`--samples 0` trong `run_baseline_benchmarks.py` có nghĩa là chấm toàn bộ manifest. Khi có `--adapter`, benchmark nạp PEFT/LoRA lên model nền đã pin revision; mặc định cũ vẫn là CPU/fp32. Các candidate suite và bake-off luôn bật `--resume-scoring`: ngay sau inference, predictions được ghi nguyên tử cùng provenance khóa SHA-256 của manifest, toàn bộ cây adapter, model/revision và tham số sinh. Nếu bước chấm CPU bị ngắt, lần chạy lại chỉ bỏ qua model/GPU khi cả specification, checksum, bytes và số dòng khớp tuyệt đối; checkpoint thiếu, cũ hoặc bị sửa sẽ fail-closed. `run_asr_candidate_suite.py` xác minh cả full ASR test và `medical_safety_asr_vi.jsonl`, đồng thời chứng minh 16 audio safety là bản sao nguyên trạng của locked test trước khi suy luận. Bộ này chấm bảo toàn cụm từ đã duyệt cho tên thuốc, liều, số, đơn vị, phủ định, thuật ngữ và code-switch; full WER, code-switch WER hoặc một slice bất kỳ không đạt đều khóa promotion. `run_mt_candidate_suite.py` áp dụng cùng nguyên tắc cho MT. Hai script lưu predictions/report/log/resource monitor, rồi tạo bundle gồm `tar.gz`, sidecar `.sha256` và manifest checksum/bytes từng file; archive được đọc lại để xác minh trước khi công bố. Kết quả fine-tune tốt nhưng gate lâm sàng fail vẫn không được promotion; 0 lỗi phát hiện được cũng không phải chứng minh an toàn.

## Điều phối GPU theo giờ

`scripts/run_gpu_rounds.py` chạy tuần tự các vòng LoRA trong `configs/gpu_rounds.yaml`, giới hạn process ở 35% VRAM và điều tiết bằng `SIGSTOP/SIGCONT`. Ngưỡng dừng VRAM cứng 40% là cấu hình bắt buộc, phải lớn hơn hoặc bằng giới hạn process và được ghi vào từng record giám sát; cấu hình NaN, sai thứ tự hoặc vượt trần bị từ chối trước khi chạy. Theo policy vận hành mới, mọi giờ đều dùng rolling GPU utilization limit 70%, resume ở 55% và hard guard phản ứng ngay từ mẫu 74% để luôn chừa biên dưới 75%; stage mới lấy mẫu mỗi 1 giây. Pilot chỉ tokenize tập ưu tiên lâm sàng đủ lớn cho số bước cấu hình (MT 8.192, ASR 6.144) để tránh lãng phí CPU. ASR pilot dùng 512 validation rows gồm cả VietMed và ViMedCSS/code-switch thay vì chọn cấu hình trên mẫu nhỏ 64 dòng. Controller chọn pilot có `eval_wer` thấp nhất, kế thừa learning rate/rank/alpha/dropout rồi tự nối vòng `final-selected-full`: toàn bộ train, 512 validation, ba epoch, effective batch 32. MT full-data dùng cấu hình đã khóa riêng. Mỗi vòng sinh log tài nguyên và `training_run.json`, rồi tạo tar.gz, sidecar SHA-256 và content manifest; bundle được đọc lại và băm từng thành viên trước khi ghi vào summary để có thể tải ngay. `scripts/watch_checkpoints.py` chạy cạnh từng round, chỉ nhận checkpoint khi `trainer_state.global_step` khớp tên thư mục và đủ model/optimizer/scheduler/RNG, sau đó tạo tar.gz nguyên tử, đọc lại cấu trúc và sinh sidecar SHA-256. Hậu kiểm candidate dùng lại chính bộ điều tiết GPU này nên không bỏ qua giới hạn tài nguyên.

`scripts/run_gpu_program.py` là state machine cho phần còn lại của Candidate A: chờ MT controller hiện tại kết thúc, xác minh selected adapter, rồi quyết định có continuation MT hay không chỉ từ bốn điểm validation loss cuối. Hệ thống chỉ kéo dài khi loss giảm đơn điệu, điểm mới nhất tốt nhất và relative gain đạt ít nhất 0,3%. Continuation warm-start từ best adapter với optimizer/scheduler mới, learning rate giảm còn `5e-5`, tối đa hai epoch bổ sung và early stopping patience 4; cách này giữ tổng ngân sách tối đa ba epoch mà không nạp lại scheduler một epoch đã cạn. Locked test tuyệt đối không được dùng để quyết định train thêm. Sau đó state machine chạy locked MT candidate suite, ASR pilots/full-data và locked ASR suite. Các GPU child luôn tuần tự; gate fail được lưu như bằng chứng và không bị nhầm với lỗi hạ tầng. Mỗi transition state được strict-serialize có byte cap rồi publish crash-durable bằng temp riêng, `fsync`, atomic replace và đọc lại; resume từ chối state/summary/trainer/gate/provenance có duplicate key, non-finite, link/junction hoặc thay đổi trong lúc đọc. Nếu một scorer MT/ASR hoặc ASR multi-round trainer hoàn tất ngoài controller, `--resume` chỉ reconcile sau khi archive, sidecar, manifest, adapter binding và từng gate/report/prediction/provenance/resource monitor đều khớp checksum. Prediction JSONL được parse strict với giới hạn tổng byte/dòng/số row và provenance exact-schema phải khóa đúng path, rows, bytes, SHA-256, task cùng toàn bộ manifest adapter hiện tại; trùng đường dẫn adapter không đủ để phục hồi. Với ASR training, mọi round archive phải hợp lệ và toàn bộ cây selected adapter sống (không chỉ `adapter_config.json`) phải được scanner dùng chung quét lặp, bounded, chống link/special-file rồi khớp chính xác danh sách file, bytes và SHA-256 trong archive; file thiếu, thừa hoặc bị sửa đều fail trước khi state thay đổi hay GPU child mới chạy. Log trong archive là nguồn bất biến nếu wrapper ghi thêm JSON terminal vào log local sau lúc đóng gói; divergence này được ghi vào state nhưng không che được sửa đổi ở artifact quan trọng. Dù cả hai Candidate A pass, `promotion_allowed` vẫn false và chuyển quyền quyết định sang vòng bake-off.

Sau Candidate A, `scripts/run_model_bakeoff.py` đóng băng adapter/checksum rồi mới chạy MT và ASR multi-model successive halving. Selection dùng hai manifest riêng sinh từ validation; test khóa cũ không được chọn model. Blind v2 chỉ mở đúng một lần sau khi winner đã chốt. Blind ASR yêu cầu tối thiểu 32 speaker và 32 recording group độc lập; scorer giữ hai định danh này trong prediction và bootstrap WER theo recording group để 95% CI không bị quá lạc quan vì nhiều câu cùng phiên ghi. Chi tiết candidate, license gate và kết quả nằm trong `docs/MODEL_BAKEOFF_RESULTS.md`.

## Backup và phục hồi

`scripts/create_verified_backup.py` tạo ba tar độc lập cho data (không lặp reports), models và reports trong thư mục staging rồi chỉ đổi tên sang đích sau khi xác minh hoàn tất. Manifest schema v2 ghi path/bytes/SHA-256 cho từng file, có checksum riêng cho manifest và `SHA256SUMS.txt` cho archive; verifier đọc lại toàn bộ nội dung tar, khóa đúng tập file và từ chối symlink, junction, special member, tên trùng, path traversal, file dư hoặc sửa nội dung dù kích thước không đổi. Archive rỗng và mọi metadata/checksum không nhất quán đều fail-closed. Backup nằm trong `.backups/`, không push lên Git vì chứa dữ liệu/model lớn và có ràng buộc license.

Bundle từ runner legacy có thể chỉ gồm archive và sidecar SHA-256. Không tự coi bundle đó là đủ điều kiện resume. `scripts/derive_legacy_evidence_manifest.py` chỉ tạo bản kê forensic `*.derived-manifest.json` sau khi xác minh sidecar, từng member, path traversal, symlink/non-file và duplicate; bản kê luôn ghi `resume_eligible=false` và không bao giờ thay thế canonical `.manifest.json`.
