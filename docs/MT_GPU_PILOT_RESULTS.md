# Kết quả pilot MT y tế trên GPU

Ngày chạy: 2026-10-04/05 (Asia/Bangkok)  
GPU: NVIDIA A100-PCIE-40GB  
Model nền: `facebook/nllb-200-distilled-600M@f8d333a098d19b4fd9a8b18f94170487ad3f821d`

## Thiết lập chung

- LoRA, bf16, batch/device 2, gradient accumulation 16, effective batch 32.
- 8.192 hàng train ưu tiên lâm sàng, oversample risk factor 2, hai chiều EN↔VI.
- 256 hàng validation, 200 optimizer steps, eval/save mỗi 50 steps.
- Process CUDA giới hạn 35% VRAM; policy áp dụng cho các stage khởi động mới là rolling 70%, resume 55% và hard reaction 74%, luôn dưới 75%. Job MT đang chạy giữ controller in-memory cũ để không mất optimizer/checkpoint.
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
| 4000 | **1.5955085754394531** | 18.741.025 | `f6377a2787af298b9d60f6258e3d2341e342d121932257e7c5c64ece0365fad5` |
| 4500 | **1.5864677429199219** | 18.741.304 | `54b4cd6bae7814dbb53aa8cb3227c2b8db79d3aa50a3a2ad10e3e38330dfd642` |
| 5000 | **1.5772212743759155** | 18.741.649 | `cbabc2dc908c7c6c2475cad621641772a5339fe375b49ac6a8825b6e6b5e018d` |
| 5500 | **1.570791482925415** | 18.743.639 | `45a0467ac9cbd8f7aec398e0db6e60b61ac36d949f4a57dd264c566857f8be06` |
| 6000 | **1.5647242069244385** | 18.745.895 | `eb09d3fd9f0ea28ff2b0c4c76b33acfa0f540d3deade915d5da3b27f1a30a971` |
| 6500 | **1.5568325519561768** | 18.747.633 | `13ff488d5cc47d5937f7c4e0aede7570c597cb90659cae3b30eec3b0eafd9d96` |
| 7000 | **1.549607276916504** | 18.748.176 | `24c0172be85ce911a4a0a6e50404646918da325a64cb7b9213d9d0aafa5188e3` |
| 7500 | **1.5457360744476318** | 18.750.238 | `dedbd421ed532caadfb22661dbfa49e070fef45260432dc125af8bad8b95b79e` |
| 8000 | **1.541010856628418** | 18.751.843 | `da3a1ce2ca998bfa19f019fd770ae2ec87ef00d9e12005b7d5893a4eeddddfdc` |
| 8500 | **1.534368634223938** | 18.754.746 | `b38f4b2660691e821b747f462552b6399380f2824c61619478a9fb8b02eea853` |
| 9000 | **1.5307471752166748** | 18.756.309 | `b0dc86b5db903bce1bb4506a9ebd9a6e9e9985e1680a7d211e35be92c8cae035` |
| 9500 | **1.5235289335250854** | 18.756.213 | `eceb106ac601cbd97d8676625ccd5ab8ceadb9bae27f54605c521edc63a1e129` |
| 10000 | **1.5207033157348633** | 18.757.992 | `8e1d871a1c0edecdb2d7b790f7b9db178007d5a0ed61eff0681bf75327777338` |
| 10500 | **1.5162816047668457** | 18.759.844 | `4382a05cc2964e219d12b532e02d18659f12bb02a542ec3bc9cc62173c4002ef` |
| 11000 | **1.5134124755859375** | 18.761.520 | `07247d950e2b5cc4a9557c1c5adb157d533112a4fb98c007c578ceb892d0cf7b` |
| 11500 | **1.509190320968628** | 18.763.341 | `d0157017fa03a7d053bd0c09775b8bcaef8fd709d3ac4c121627524978d5827b` |
| 12000 | **1.505143165588379** | 18.764.962 | `c225df97ff9d56e260ddf5185f8c94e171ffa9e170295e3f34d049cad21d5290` |
| 12500 | **1.5013389587402344** | 18.765.448 | `b68b11f4ae6aa69d87bcc98214e6c0b5e43b0d86aa8aec0b23216eede3b07145` |
| 13000 | **1.4996998310089111** | 18.767.067 | `b295797f437e288ebbb66f7a714c36cf0983b11dcbe7aba08ba4858dea8885b2` |
| 13500 | **1.4964866638183594** | 18.768.849 | `19efe8a6d3337635c33268fcb38f8919547c7d23dcd6655edc5c79769455c253` |
| 14000 | **1.4928605556488037** | 18.769.225 | `01080eabbc5d460d5a45bceef88d9570b4b0e6aab77b7a16c0ebf333ff86bfe1` |
| 14500 | **1.4911822080612183** | 18.770.579 | `a5a1823f04c4a08b40ed4cde2406597dd2fea9490cfbdcc3e6fe52a6409a1f1f` |
| 15000 | **1.4896564483642578** | 18.771.899 | `ddddd76e68ee03be7b4d1f7baacca13225b6dfcff54127863acf5195d0ba8b81` |
| 15500 | **1.4854638576507568** | 18.772.820 | `42855cb444b662809282de445ac858b10d9d08d058dfc75fd1601c4b18e02249` |
| 16000 | **1.4826416969299316** | 18.773.509 | `3502aa3ace7143d9f49a3f8c0739241d7ee4e139e4ad0fc22e89c593d3d57576` |
| 16500 | **1.4798743724822998** | 18.775.961 | `5cbb3fb07a4f04ea96f353ffd616d2492e05a2dae49e06f88bcbfd75193c83a4` |
| 17000 | **1.4781849384307861** | 18.775.845 | `c3b55c16e91c14e690bff3aeb19fa8bec0b986003fe26813d2c9e02baf92ffa9` |
| 17500 | **1.4769376516342163** | 18.777.804 | `e653c6df775bdfc7d87d1f844dae95c51da49bd7d98b449bd76ec121617abcd2` |
| 18000 | **1.4739108085632324** | 18.779.083 | `44d9ef037cdf3bed9299d1335afab24222246b42c94aedc83056bdf2aabdac71` |
| 18500 | **1.4733152389526367** | 18.780.212 | `c36bcbe5dac0ff1aa08b279511f48bddce9856e502b196394057d6c4d152ba64` |
| 19000 | **1.4706478118896484** | 18.780.194 | `e5347ecb6b54360e58290e8be2dc07acd649bbe38623cefff3b23b1f977374b6` |

Ba mươi tám archive, sidecar checksum và manifest đã được tải về `test/checkpoints/mt-20261004-172023/` rồi băm lại trên local. Các checkpoint mới còn được đọc lại cấu trúc để xác nhận có adapter, optimizer, scheduler, RNG và trainer state. Eval loss sớm đang giảm nhưng chưa phải BLEU/chrF trên test khóa và không phải bằng chứng an toàn; promotion vẫn bị khóa tới khi full-run, full test và toàn bộ clinical gate hoàn tất.
