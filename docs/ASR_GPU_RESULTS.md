# ASR Candidate A — kết quả GPU và trạng thái đánh giá

## Full training

PhoWhisper-small Candidate A đã hoàn tất bốn vòng tuần tự trong run `asr-20261005-222621`. Pilot `r32/lr2e-5` được chọn; full training dùng LoRA rank 32, alpha 64, dropout 0,1, 3 epoch, batch 2 và gradient accumulation 16. Mọi child dùng giới hạn 35% VRAM/process, hard memory 40%, rolling utilization 70% và hard reaction 74%.

| Vòng | Eval WER | Archive SHA-256 |
| --- | ---: | --- |
| pilot r8/lr1e-4 | 43,1929 | `c549362511afc797e52369965787af5b504c7115648c01d6ab7ff9d353011533` |
| pilot r16/lr5e-5 | 33,9176 | `cc874a1697ebfea983c71b0b2fcad9b4e97390bb3fd64ec63e79c72e2ae7270d` |
| pilot r32/lr2e-5 | 31,2608 | `a6ee1d319328b5204734d10f050a2dbc0b1d2b790f5e9a9d0fb0f581c9fa80f1` |
| full selected | 28,5726 | `ae6168db87a465a8baa2ef3de8237738269a2ad36df9b7f33349cb1bc7e38240` |

Full run đạt 2.871/2.871 bước. Checkpoint tốt nhất theo WER là bước 2.750; checkpoint kết thúc 2.871 giữ cùng eval WER. Tất cả 12 checkpoint (250–2.871), bốn archive vòng train, sidecar và content manifest đã được tải về local và xác minh lại SHA-256, byte size, đường dẫn an toàn và hash từng member.

## Locked candidate evaluation

Lần chạy tự động đầu tiên kết thúc với trạng thái hạ tầng `error`, **không phải clinical fail**. Whisper encoder được nạp `bfloat16` nhưng benchmark đưa `input_features` `float32`, gây lỗi `Input type (torch.cuda.FloatTensor) and weight type (CUDABFloat16Type) should be the same` trước prediction đầu tiên. Vì vậy chưa có WER locked, code-switch WER hay kết quả riêng cho thuốc/liều/số/đơn vị/phủ định từ lần chạy này.

Benchmark hiện ép feature tensor theo dtype thực của convolution đầu vào encoder, kể cả khi PEFT giữ tham số LoRA ở fp32. Regression test mô phỏng PEFT wrapper khóa hành vi này. Candidate suite phải được chạy lại trọn vẹn trước bake-off; chỉ report có predictions, provenance và clinical gate hợp lệ mới được dùng làm Candidate A reference. Fine-tune và eval loss/WER validation không tự chứng minh an toàn.
