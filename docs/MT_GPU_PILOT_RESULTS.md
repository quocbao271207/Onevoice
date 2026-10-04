# Kết quả pilot MT y tế trên GPU

Ngày chạy: 2026-10-04/05 (Asia/Bangkok)  
GPU: NVIDIA A100-PCIE-40GB  
Model nền: `facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d`

## Thiết lập chung

- LoRA, bf16, batch/device 2, gradient accumulation 16, effective batch 32.
- 8.192 hàng train ưu tiên lâm sàng, oversample risk factor 2, hai chiều EN↔VI.
- 256 hàng validation, 200 optimizer steps, eval/save mỗi 50 steps.
- Process CUDA giới hạn 35% VRAM; rolling utilization ban ngày 38%, 02:00–09:00 Asia/Bangkok 75%.
- Runtime monitor đã tự chuyển `active_utilization_limit_percent` từ 38 sang 75 và `boosted_window` từ false sang true lúc `2026-10-05T02:00:13+07:00`, không restart trainer.
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

## Full-run đang chạy

Run `mt-20261004-172023/01-final-r8-lr1e4-full` dùng toàn bộ 339.028 cặp train, hai chiều và oversample lâm sàng hệ số 2, tổng 33.521 optimizer steps. Watcher checkpoint nguyên tử được bổ sung giữa run và đã chứng minh bắt được checkpoint mới, không chỉ xử lý checkpoint tồn tại sẵn.

| Checkpoint | Eval loss trên 2.048 mẫu hai chiều | Archive bytes | SHA-256 |
|---:|---:|---:|---|
| 500 | 1.8476485013961792 | 18.777.533 | `3a483c3d8f0e57c6dc6ec94469c89da0ad23478797fb7d61625a0318fa492021` |
| 1000 | 1.7338323593139648 | 18.755.502 | `3bfca366b1a8be5262aecb51bf8676e413e8bfd1ad22bd8aae947cdb83a8a66b` |
| 1500 | 1.6891976594924927 | 18.743.891 | `40501d03c9115b473e228b30c3f148c8bdb2b4215bf658a25710cad784d737b5` |
| 2000 | **1.659508228302002** | 18.742.104 | `1cda28d6382e15ddf1f3fa879ff3f69650d5c5073c95b6e4f2f50fa5e66be0d1` |
| 2500 | **1.6377246379852295** | 18.741.268 | `f098ef846d13b8c7e85a906b8ab7bbee6a8738700fd61d05b7f7cadf8c5869c6` |
| 3000 | **1.6207243204116821** | 18.740.534 | `248957850ca4413c3afebd3af0ac348f8cdef60f37dff9b27d85a4283a70b427` |
| 3500 | **1.6084938049316406** | 18.740.473 | `069d95c3d7080876e2e030d860a3c8e38beea78f31f8e8bbdacc53c61ff39e04` |

Bảy archive, sidecar checksum và manifest đã được tải về `test/checkpoints/mt-20261004-172023/` rồi băm lại trên local. Các checkpoint mới còn được đọc lại cấu trúc để xác nhận có adapter, optimizer, scheduler, RNG và trainer state. Eval loss sớm đang giảm nhưng chưa phải BLEU/chrF trên test khóa và không phải bằng chứng an toàn; promotion vẫn bị khóa tới khi full-run, full test và toàn bộ clinical gate hoàn tất.
