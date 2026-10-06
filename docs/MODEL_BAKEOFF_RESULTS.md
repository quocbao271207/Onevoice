# Multi-model bake-off — trạng thái và kết quả

Ngày mở vòng: 05/10/2026.

## Trạng thái hiện tại

Chưa model bổ sung nào được gọi là `benchmarked`. Candidate A vẫn đang được tạo bởi pipeline hiện hữu và không bị restart hay thay đổi process đang chạy:

- MT Candidate A: `facebook/nllb-200-distilled-600M`, full-data LoRA đã hoàn tất 33.521 bước; candidate suite kết thúc ngày 06/10/2026 với `status=fail`, `error=null`. Hard gate phát hiện lỗi liều 11,11%, số 8,33%, phủ định 5,56%, thuật ngữ 27,78% và code-switch 50%; model không được promotion.
- ASR Candidate A: `vinai/PhoWhisper-small`; ba pilot 150 bước đã hoàn tất và chọn LoRA r32 theo `eval_wer=31,2608` (r16: 33,9176; r8: 43,1929). Full-data adaptive run đang tiếp tục tuần tự; không có controller hoặc GPU child trùng được khởi động.
- Các challenger đều là `planned`; `data/reports/model_bakeoff/comparison.json` là nguồn trạng thái máy đọc được.
- Candidate A đạt gate cũ vẫn chỉ được đóng băng làm chuẩn tham chiếu. `promotion_allowed` luôn là `false` cho đến khi hoàn tất bake-off, blind v2 và deployment gate.

Artifact MT legacy đã được tải về `test/program-20261005-014500`: archive 3.938.906 byte có SHA-256 `d908ddc739acb9067514b0139a291bb67c91c465b7f096418064ebc5bf83cf81`, khớp sidecar gốc. Runner cũ không phát hành canonical `.manifest.json` và archive thiếu provenance cho full predictions, nên artifact này không đủ điều kiện resume/promotion theo runner mới. Bản kê `mt-candidate.tar.gz.derived-manifest.json` chỉ xác minh forensic 12 member/22.861.924 byte và luôn ghi `resume_eligible=false`; `download_manifest.json` khóa lại toàn bộ file đã tải. Đây vẫn chỉ là một phần chương trình: ASR Candidate A, challenger training, blind v2 và deployment benchmark còn ở phía sau.

Checkpoint ASR full-data bước 500 và 750 đã được watcher phát hành, tải về local và xác minh ngày 06/10/2026. Archive bước 750 có 28.888.504 byte, SHA-256 `b565ccfd1c8602bbc71a720b63a3bee13a65f9e9f78f7124657b7002485aabd8`; sidecar, content manifest và checkpoint index khớp. Cả 15/15 regular member (34.773.485 byte nội dung) đều đúng hash, không có path traversal, duplicate, symlink hay special member; `trainer_state.global_step=750` và `max_steps=2871`.

`eval_loss` tiếp tục giảm từ 1,5151 ở bước 250 qua 1,1474 ở bước 500 xuống 0,9435 ở bước 750. Tuy nhiên `eval_wer` giảm từ 67,3243 xuống 39,1998 rồi tăng nhẹ lên 39,9387, nên best checkpoint vẫn là bước 500 theo policy `eval_wer` lower-is-better; không được đổi winner chỉ vì loss thấp hơn. Đây là validation trend, không phải locked-test safety evidence và chưa cho phép promotion.

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
- Model selection: `data/eval/mt_selection_dev.jsonl` (256 cặp) và `data/eval/asr_selection_dev.jsonl` (384 câu), chỉ sinh từ validation. Bake-off preflight tự tính lại fingerprint từ payload, khóa uniqueness và kiểm tra ID/cặp dịch/transcript/audio hash/speaker/recording group không giao cả train lẫn mọi locked test hay safety manifest trước khi dùng GPU.
- Blind locked test v2: `data/eval/blind_test_v2.lock.json` hiện là `awaiting_unseen_data`. Không tự chế blind set từ dữ liệu model đã nhìn thấy. Schema v2 yêu cầu tối thiểu **256 cặp MT** và **384 clip ASR** thật sự unseen; các hàng có thể thuộc nhiều lát nhưng không được nhân bản để tăng đếm.
- Test/safety cũ không được mở lặp lại để chọn model.

Selection dev chứa lát thuốc, liều, số, đơn vị, phủ định, thuật ngữ và code-switch. ASR còn giữ accent/role/recording-condition để báo riêng. Blind v2 chỉ được khóa khi mỗi cặp MT dùng được cho cả hai chiều và đạt 256 mẫu/chiều. MT phải dùng schema chuẩn `source_language=en`, `target_language=vi`; ASR phải khai báo ngôn ngữ Việt, kể cả `vi-code-switch`, cùng `speaker` và `group` ổn định để kiểm tra speaker/recording-disjoint. Mỗi lát thuốc, liều, số, đơn vị, phủ định, thuật ngữ và code-switch phải có ít nhất 32 mẫu cho từng task. ASR còn bắt buộc ít nhất 64 clip Bắc, 64 Trung, 64 Nam, 64 bác sĩ, 64 bệnh nhân, 96 clip nhiễu, **32 speaker độc lập và 32 recording group độc lập**; các thuộc tính có thể giao nhau trên cùng clip. Script lưu lại thống kê coverage trong lock, tính lại khi verify và fail-closed nếu thiếu quota, thiếu ID/fingerprint, có ID/pair/audio trùng trong chính blind set, hoặc có ID, text fingerprint, pair fingerprint, audio SHA, speaker hay recording group giao train/selection/test cũ. Audit không tin rằng các tập lịch sử đã có sẵn fingerprint: nó tự tính lại pair/text fingerprint từ payload và tự băm file audio khi SHA bị thiếu, nên safety MT/ASR cũ không thể bị né bằng cách đổi ID. Prediction ASR giữ `speaker/group`; WER 95% CI bootstrap theo recording group thay vì coi các câu cùng buổi ghi là độc lập. Report sau inference phải khớp chính xác số hàng, số mẫu từng lát, số speaker/group và số cluster trong lock. Metadata khai báo sai, file audio thiếu hoặc bị sửa đều bị từ chối.

Blind v2 không chỉ kiểm tra safety. Nó dùng ngưỡng release đã khóa SHA-256 trong `configs/model_bakeoff.yaml` và lấy nội dung từ `configs/accuracy_program.yaml`: cận trên 95% CI của ASR WER phải không vượt ngưỡng, còn cận dưới 95% CI của MT BLEU và chrF2 phải đạt ngưỡng cho từng chiều. Preflight và blind runner đều từ chối nếu checksum policy thay đổi. CER và code-switch WER cũng phải hữu hạn, code-switch WER phải đạt policy. Thiếu metric/CI, NaN hoặc point estimate/CI không đạt đều khóa promotion.

Mỗi slot blind chỉ được mở cho winner đúng task/chiều trong snapshot `selection_comparison-<identity>.json` bất biến; tên snapshot lấy từ digest của winner bindings nên các vòng bake-off sau không ghi đè vòng cũ. `comparison.json` có thể tiếp tục nhận blind/deployment result mà không làm hỏng checksum selection đã khóa. Trước khi suy luận, runner khóa SHA-256 của snapshot này, manifest cây adapter và blind manifest vào specification; adapter bị sửa tại cùng đường dẫn cũng bị từ chối. Report có provenance sidecar chứa checksum/bytes và toàn bộ binding này, nên resume chỉ được dùng lại report đã xác minh; report có sẵn nhưng thiếu provenance hoặc bị sửa sẽ fail-closed thay vì được chấm lại.

## Successive halving và winner gate

Mỗi GPU child chạy tuần tự, 35% VRAM/process, hard memory 40%, rolling utilization 70%, resume 55% và hard reaction 74% — luôn dưới 75%. Stage mới lấy mẫu tài nguyên mỗi 1 giây; preflight từ chối ngưỡng không hữu hạn, sai thứ tự hoặc vượt trần trước khi khởi chạy child.

Selection comparison phải chứa checksum của policy đa metric/95% CI và toàn bộ input selection-dev + accuracy policy. Blind runner từ chối comparison cũ, thiếu policy hoặc lệch checksum; vì vậy kết quả do waiter sống lâu nạp từ commit cũ chỉ là provisional. Sau khi waiter đó kết thúc ở trạng thái `waiting_for_blind_test_v2`, runner hiện hành phải resume cùng state directory để tái dùng artifact hợp lệ, bổ sung stage còn thiếu và phát hành immutable selection snapshot mới trước khi blind test được phép mở.

1. Zero-shot trên cùng selection dev.
2. Pilot 400 steps, effective batch 32.
3. Loại candidate fail safety hoặc thua rõ; giữ tối đa hai.
4. Semifinal 2.000 steps từ cùng base revision.
5. Full train giữ các candidate không bị candidate dẫn đầu Pareto-dominance theo 95% CI; một metric tốt không được che một metric khác kém rõ rệt.
6. MT EN→VI và VI→EN được chọn độc lập.

So sánh theo effective train samples, không theo thời gian. Safety là hard gate: thuốc/liều/số/đơn vị/phủ định phải 0 failure; terminology và code-switch cũng có policy riêng. Sau safety, MT bắt buộc đủ cả BLEU và chrF2 theo chiều cùng bootstrap 95% CI; ASR bắt buộc đủ WER, CER và code-switch WER, trong đó WER có bootstrap 95% CI. Chỉ thay Candidate A khi ít nhất một metric có CI tốt hơn rõ rệt và không metric có CI nào thụt lùi rõ rệt; CER/code-switch vẫn phải hữu hạn và code-switch phải qua policy riêng. Winner search duyệt toàn bộ challenger đã qua safety theo thứ hạng và chọn challenger cao nhất thực sự thỏa luật đa metric; một model đứng đầu point estimate nhưng có CI regression không được che mất model xếp sau có Pareto improvement hợp lệ.

Candidate A MT cũng được benchmark độc lập theo từng chiều; lỗi an toàn hoặc khoảng tin cậy của EN→VI không được làm thay đổi quyết định VI→EN và ngược lại. Metric hay bootstrap 95% CI thiếu, không hữu hạn hoặc đảo cận đều bị loại fail-closed trước ranking.

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

Runner có resume theo command digest, checkpoint watcher, archive/checksum từng round và từ chối chạy hai command khác nhau dưới cùng stage. Mỗi file đầu ra của stage còn được khóa path/bytes/SHA-256 trong state; file bị thay thế sau khi hoàn tất sẽ làm resume fail-closed. Benchmark do waiter cũ hoàn tất nhưng chưa có khóa output sẽ tự chạy lại đúng đường CPU scoring từ prediction checkpoint đã xác minh bằng `--resume-scoring`, không nạp model hoặc suy luận GPU lần nữa. Trước khi bất kỳ adapter bake-off nào được benchmark, runner đọc lại tar, sidecar và content manifest của mọi round rồi đối chiếu toàn bộ cây adapter sống theo path/bytes/SHA-256; chỉ có `adapter_config.json` là không đủ. Manifest freeze/winner dùng đường dẫn POSIX ổn định giữa local và GPU, đồng thời từ chối symlink ở root hoặc bất kỳ file con nào. Khi blind v2 chưa sẵn sàng, runner dừng ở trạng thái chờ thay vì tự mở test cũ.

## Deployment reference

Dự án giữ immutable evidence QCS6490 đã đo ngày 26/09/2026. Tuy nhiên trang Qualcomm hiện tại được kiểm tra lại ngày 05/10/2026 không còn liệt kê QCS6490 cho Whisper-Small-Quantized. Vì vậy evidence cũ không bị xóa, nhưng promotion mới bắt buộc rerun trên board QCS6490 và báo latency p50/p95, peak RAM/VRAM, model bytes. Không suy diễn khả năng deployment của winner fine-tuned chỉ từ reference OpenAI Whisper-small.

`deployment_selected_winners.json` phải là schema version 1, ghi `measurement_source=physical_board`, timestamp có timezone và metadata board/chipset/OS. Metadata tự khai không đủ: phải chạy `scripts/capture_qcs6490_identity.py` ngay trên board, giữ JSON device-tree/sysfs bất biến dưới `data/reports/model_bakeoff/board-evidence/`, và để finalizer khóa SHA-256. Gate yêu cầu ARM64 cùng cả model board và chuỗi `qcom` compatible nhận dạng QCS6490/RB3 Gen 2; timestamp identity phải nằm trong 24 giờ của phiên benchmark. Arduino, cloud host, evidence cũ hoặc file ngoài thư mục evidence đều bị từ chối. Mỗi winner phải khớp task, chiều, candidate ID và SHA-256 manifest của đúng adapter; artifact triển khai phải nằm dưới `models/`, có path/SHA-256/byte size khớp file thật. Report phải chứa ít nhất 30 latency samples để runner tự tính lại p50/p95; p95 không được thấp hơn p50 và mọi số đo phải hữu hạn/dương (VRAM được phép bằng 0 nếu chạy NPU/CPU). Report cloud cũ, report của model khác hoặc số đo giả/thiếu đều bị reject.

Không soạn report cuối bằng tay. Khi blind v2 pass, state machine tự tạo draft đã khóa đúng ba winner, checksum cây adapter và checksum `comparison.json`; lệnh tương đương để phục hồi thủ công là:

```bash
python scripts/prepare_deployment_benchmark.py --action template \
  --selection-comparison data/reports/model_bakeoff/comparison.json
```

CLI chỉ tạo draft sau khi `comparison.json` đã ở trạng thái `blind_complete`. Trên QCS6490 thật, chạy `python scripts/capture_qcs6490_identity.py --output data/reports/model_bakeoff/board-evidence/qcs6490-identity.json`, rồi chép file evidence và artifact thực tế về đúng các đường dẫn tương ứng trong project. Điền `measured_at`, `device.os`, `device.identity_evidence_path`, `artifact_path`, ít nhất 30 `latency_samples_ms`, `peak_ram_bytes` và `peak_vram_bytes` vào file `.draft.json`; không tự điền `device.board` hoặc checksum identity vì finalizer lấy trực tiếp từ evidence. Sau đó chạy `--action finalize` với cùng comparison. CLI tự tính p50/p95, số lượt đo, SHA-256/kích thước từng artifact và chỉ tạo `deployment_selected_winners.json` nếu toàn bộ physical-board gate pass; draft/report có sẵn không bị ghi đè.
