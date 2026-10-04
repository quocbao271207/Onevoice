# Kết quả pilot MT y tế trên GPU

Ngày chạy: 2026-10-04/05 (Asia/Bangkok)  
GPU: NVIDIA A100-PCIE-40GB  
Model nền: `facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d`

## Thiết lập chung

- LoRA, bf16, batch/device 2, gradient accumulation 16, effective batch 32.
- 8.192 hàng train ưu tiên lâm sàng, oversample risk factor 2, hai chiều EN↔VI.
- 256 hàng validation, 200 optimizer steps, eval/save mỗi 50 steps.
- Process CUDA giới hạn 35% VRAM; rolling utilization ban ngày 38%, 02:00–09:00 Asia/Bangkok 75%.
- Cả ba vòng có 0 lần throttle và không vượt giới hạn đã cấu hình.

## So sánh

| Round | LoRA | Learning rate | Best eval loss | Train loss | Archive bytes | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| `pilot-r8-lr1e4` | r8/α16 | 1e-4 | **1.8115464448928833** | 1.9591863346099854 | 47.629.430 | `1956972aa76269288484ca15737baf813b5d83dd8b5894666ce0a1c8a2f55a28` |
| `pilot-r16-lr5e5` | r16/α32 | 5e-5 | 1.8412916660308838 | 1.9014 | 77.953.821 | `11517996fc385fa9d72d68fbce3c238d767ce5d923ebf9e33c4977e46de8e1a0` |
| `pilot-r32-lr2e5` | r32/α64 | 2e-5 | 1.8772460222244263 | 1.9991625690460204 | 138.466.707 | `cf3c890e556525a5144621aa9c84ab3383dd31aeec21968c82735b2cae70f71f` |

## Quyết định

Chọn `r8/α16, lr=1e-4` cho full-run vì có validation loss thấp nhất, adapter nhỏ nhất và chi phí thấp nhất. `configs/gpu_rounds.yaml` có round `final-r8-lr1e4-full`: dùng toàn bộ train, một epoch, batch/device 4 và accumulation 8 (effective batch vẫn 32), validation giữa kỳ 1.024 hàng để kiểm soát thời gian. Cấu hình 4×8 thay thế 2×16 sau probe đầu full-run vì 2×16 chỉ dùng khoảng 15–23% GPU và tạo ETA hơn 30 giờ. Sau huấn luyện phải chạy locked validation/test đầy đủ và suite lâm sàng riêng cho tên thuốc, liều lượng, số, đơn vị, phủ định và code-switch. Fine-tune không phải bằng chứng an toàn và checkpoint không được promotion nếu các gate này chưa đạt.

Ba archive đã được tải về `test/` ở máy local và xác minh lại SHA-256.

## Hậu kiểm đã chuẩn bị

`scripts/run_mt_candidate_suite.py` sẽ chạy ngay sau full-run. Script nạp adapter LoRA trên đúng NLLB revision đã pin, chấm toàn bộ `mt--test.jsonl` theo cả hai chiều, sau đó chấm riêng suite `medical_safety_mt.jsonl` đã xác minh checksum. Candidate chỉ pass khi BLEU/chrF đạt ngưỡng và đủ cả bảy slice `drug_name`, `dose`, `number`, `unit`, `negation`, `terminology`, `code_switch` với failure rate theo policy. Predictions, report, log và resource trace được đóng gói thành tar.gz kèm SHA-256 để tải về local ngay.
