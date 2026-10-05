# Multi-model bake-off — trạng thái và kết quả

Ngày mở vòng: 05/10/2026.

## Trạng thái hiện tại

Chưa model bổ sung nào được gọi là `benchmarked`. Candidate A vẫn đang được tạo bởi pipeline hiện hữu và không bị restart hay thay đổi process đang chạy:

- MT Candidate A: `facebook/nllb-200-distilled-600M`, full-data LoRA đã hoàn tất 33.521 bước; candidate suite đang chạy.
- ASR Candidate A: `vinai/PhoWhisper-small`, chờ candidate suite MT hoàn tất trong state machine hiện hữu.
- Các challenger đều là `planned`; `data/reports/model_bakeoff/comparison.json` là nguồn trạng thái máy đọc được.
- Candidate A đạt gate cũ vẫn chỉ được đóng băng làm chuẩn tham chiếu. `promotion_allowed` luôn là `false` cho đến khi hoàn tất bake-off, blind v2 và deployment gate.

Tại lần chụp trạng thái gần nhất ngày 05/10/2026, MT đã hoàn tất 33.521/33.521 bước. Toàn bộ 68 checkpoint archive, archive cuối và summary đã có checksum xác minh trên local; `program_state.json` đã chuyển sang `mt_candidate`. Đây chỉ là hoàn tất huấn luyện MT Candidate A, không phải hoàn tất chương trình: ASR Candidate A, challenger training, blind v2 và deployment benchmark còn ở phía sau. Tiến độ end-to-end ước khoảng 35%; đây là ước lượng theo stage, không phải accuracy.

## Candidate và license gate

| Candidate | Vai trò | License quan sát | Trạng thái GPU mặc định |
|---|---|---|---|
| NLLB-600M | MT Candidate A | CC-BY-NC-4.0 | Được chạy trong phạm vi nghiên cứu, không đủ gate production |
| VinAI Translate EN→VI v2 | MT research challenger | AGPL-3.0 | Khóa; cần phê duyệt research-license rõ ràng trước khi tốn GPU |
| VinAI Translate VI→EN v2 | MT research challenger | AGPL-3.0 | Khóa; cần phê duyệt research-license rõ ràng trước khi tốn GPU |
| M2M100-418M | MT production fallback | MIT | Qua license gate; được chọn thay mBART-50 vì nhỏ hơn |
| PhoWhisper-small | ASR Candidate A | BSD-3-Clause | Qua gate |
| OpenAI Whisper-small multilingual | ASR challenger | Apache-2.0 | Qua gate |
| PhoWhisper-base | ASR nhẹ | BSD-3-Clause | Qua gate |
| Qualcomm Whisper-Small-Quantized | Deployment reference | Apache-2.0 | Chỉ đo deployment, không fine-tune |

License metadata được pin cùng revision trong `configs/model_bakeoff.yaml`. Đây là gate kỹ thuật fail-closed, không thay thế tư vấn pháp lý. VinAI chỉ được mở bằng `--approve-research-license <candidate-id>`; cờ này không bao giờ đổi `production_eligible` thành true.

## Dữ liệu công bằng

- Train: giữ nguyên manifest train đã audit.
- Model selection: `data/eval/mt_selection_dev.jsonl` (256 cặp) và `data/eval/asr_selection_dev.jsonl` (384 câu), chỉ sinh từ validation và đã kiểm tra không giao với test.
- Blind locked test v2: `data/eval/blind_test_v2.lock.json` hiện là `awaiting_unseen_data`. Không tự chế blind set từ dữ liệu model đã nhìn thấy.
- Test/safety cũ không được mở lặp lại để chọn model.

Selection dev chứa lát thuốc, liều, số, đơn vị, phủ định, thuật ngữ và code-switch. ASR còn giữ accent/role/recording-condition để báo riêng. Blind v2 chỉ được khóa khi có đủ hai chiều MT và ASR Bắc/Trung/Nam, bác sĩ/bệnh nhân và nhiễu. Script fail-closed nếu thiếu ID/fingerprint, có ID/pair/audio trùng trong chính blind set, hoặc có ID, text fingerprint, pair fingerprint hay audio SHA giao train/selection/test cũ.

## Successive halving và winner gate

Mỗi GPU child chạy tuần tự, 35% VRAM/process, hard memory 40%, rolling utilization 70%, resume 55% và hard reaction 74% — luôn dưới 75%.

1. Zero-shot trên cùng selection dev.
2. Pilot 400 steps, effective batch 32.
3. Loại candidate fail safety hoặc thua rõ; giữ tối đa hai.
4. Semifinal 2.000 steps từ cùng base revision.
5. Full train chỉ winner hoặc candidate có 95% CI giao winner.
6. MT EN→VI và VI→EN được chọn độc lập.

So sánh theo effective train samples, không theo thời gian. Safety là hard gate: thuốc/liều/số/đơn vị/phủ định phải 0 failure; terminology và code-switch cũng có policy riêng. Sau safety, MT dùng BLEU/chrF2 theo chiều với bootstrap 95% CI; ASR dùng WER/CER/code-switch WER và 95% CI. Chỉ thay Candidate A khi khoảng tin cậy chứng minh challenger tốt hơn.

## Chứng cứ cũ đã sửa

`data/reports/experiments/decoding/mt_val_greedy_rescored.json` dùng prediction của NLLB greedy. File đã được gắn `model=facebook/nllb-200-distilled-600M` và `evidence_status=invalid_for_model_comparison`; không còn là bằng chứng VinAI Translate. NLLB greedy/beam gốc vẫn hợp lệ vì có prediction, model ID và revision rõ ràng.

## Lệnh vận hành

```bash
python scripts/build_selection_dev.py
python scripts/run_model_bakeoff.py --preflight

# Chỉ chạy sau khi current program state complete. Process này có thể chờ mà không dùng GPU.
python scripts/run_model_bakeoff.py \
  --execute --wait-current \
  --current-program-state gpu-runs/program-YYYYMMDD-HHMMSS/program_state.json \
  --state-dir gpu-runs/model-bakeoff

# Khi có blind v2 thật sự chưa từng bị model nhìn thấy:
python scripts/run_blind_candidate_suite.py --action lock \
  --mt-manifest /secure/blind_v2_mt.jsonl \
  --asr-manifest /secure/blind_v2_asr.jsonl
```

Runner có resume theo command digest, checkpoint watcher, archive/checksum từng round và từ chối chạy hai command khác nhau dưới cùng stage. Khi blind v2 chưa sẵn sàng, nó dừng ở trạng thái chờ thay vì tự mở test cũ.

## Deployment reference

Dự án giữ immutable evidence QCS6490 đã đo ngày 26/09/2026. Tuy nhiên trang Qualcomm hiện tại được kiểm tra lại ngày 05/10/2026 không còn liệt kê QCS6490 cho Whisper-Small-Quantized. Vì vậy evidence cũ không bị xóa, nhưng promotion mới bắt buộc rerun trên board QCS6490 và báo latency p50/p95, peak RAM/VRAM, model bytes. Không suy diễn khả năng deployment của winner fine-tuned chỉ từ reference OpenAI Whisper-small.
