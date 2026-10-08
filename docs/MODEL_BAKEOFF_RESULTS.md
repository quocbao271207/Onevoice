# Multi-model bake-off — trạng thái và kết quả

Ngày mở vòng: 05/10/2026.

## Trạng thái hiện tại

Candidate A đã hoàn tất training nhưng fail clinical gate và chỉ được giữ làm reference. Các profile M2M100 đã hoàn tất cũng đều fail clinical gate; pilot VI→EN r32 bị gián đoạn sau checkpoint 100 nên bake-off chưa có winner:

- MT Candidate A: `facebook/nllb-200-distilled-600M`, full-data LoRA đã hoàn tất 33.521 bước; candidate suite kết thúc ngày 06/10/2026 với `status=fail`, `error=null`. Hard gate phát hiện lỗi liều 11,11%, số 8,33%, phủ định 5,56%, thuật ngữ 27,78% và code-switch 50%; model không được promotion.
- ASR Candidate A: `vinai/PhoWhisper-small`; ba pilot 150 bước đã hoàn tất và chọn LoRA r32 theo `eval_wer=31,2608` (r16: 33,9176; r8: 43,1929). Full-data đã hoàn tất 2.871/2.871 bước; best checkpoint 2.750 có `eval_wer=28,5726`. Rerun locked evaluation bằng evaluator đã sửa dtype kết thúc với `status=fail`, `error=null`: WER tổng 28,0009%, code-switch WER 26,3934%; hard gate phát hiện lỗi tên thuốc 92,31%, liều 100%, số 77,78%, đơn vị 77,78%, phủ định 16,67%, thuật ngữ 62,5% và code-switch clinical 87,5%. Model không được promotion nhưng được giữ làm reference để challenger phải vượt qua.
- M2M100-418M EN→VI zero-shot đạt BLEU 30,1245 (95% CI 27,6601–32,4378), chrF2 48,7316 (95% CI 46,5896–50,5581), safety failure 25,78% và fail clinical gate. Pilot r8/lr1e-4 sau 400 bước đạt BLEU 31,9013 (95% CI 29,0378–34,5290), chrF2 48,7526 (95% CI 46,0748–51,1436), nhưng safety failure tăng lên 27,34% và vẫn fail gate. Chênh lệch BLEU có CI chồng lấn, chrF2 gần như không đổi; vì vậy profile này chưa được gọi là mạnh hơn và không được promotion.
- Profile M2M100 EN→VI r16/lr5e-5 đạt BLEU 31,2398 (95% CI 28,0574–33,7972), chrF2 48,1408 (95% CI 45,5707–50,5991), safety failure 27,73%; r32/lr2e-5 đạt BLEU 31,4021 (95% CI 28,6987–33,8243), chrF2 48,4826 (95% CI 46,0739–50,6603), safety failure 27,73%. Cả hai fail clinical gate, không vượt r8 rõ rệt và không cải thiện safety nên không promotion. M2M100 VI→EN zero-shot đạt BLEU 24,5647 (95% CI 22,2407–26,9165), chrF2 50,7386 (95% CI 49,2193–52,2650), safety failure 25,78% và fail gate. Pilot r8 VI→EN đạt BLEU 25,0337 (95% CI 22,4293–27,8237), chrF2 50,0401 (95% CI 47,8307–52,0870), safety failure 26,95%; r16 đạt BLEU 24,5788 (95% CI 21,7107–27,4356), chrF2 50,0091 (95% CI 47,8453–52,0338), safety failure 25,78%. Cả hai fail gate, không vượt zero-shot ngoài khoảng bất định và không được promotion. Pilot r32 VI→EN kết thúc bất thường gần bước 110/400: stage `return_code=1`, monitor `return_code=-9`, không có controller `hard_stop_reason`/`stop_reason`, không có kernel OOM evidence và không có checkpoint 200. Vì vậy lần chạy này không phải benchmark hoàn tất hoặc promotion evidence. Các challenger ASR Whisper-small multilingual rồi PhoWhisper-base vẫn chờ chạy tuần tự. `data/reports/model_bakeoff/comparison.json` chỉ trở thành nguồn winner máy đọc được sau khi các round tương ứng hoàn tất.
- Candidate A đạt gate cũ vẫn chỉ được đóng băng làm chuẩn tham chiếu. `promotion_allowed` luôn là `false` cho đến khi hoàn tất bake-off, blind v2 và deployment gate.

Artifact MT legacy đã được tải về `test/program-20261005-014500`: archive 3.938.906 byte có SHA-256 `d908ddc739acb9067514b0139a291bb67c91c465b7f096418064ebc5bf83cf81`, khớp sidecar gốc. Runner cũ không phát hành canonical `.manifest.json` và archive thiếu provenance cho full predictions, nên artifact này không đủ điều kiện resume/promotion theo runner mới. Bản kê `mt-candidate.tar.gz.derived-manifest.json` chỉ xác minh forensic 12 member/22.861.924 byte và luôn ghi `resume_eligible=false`; `download_manifest.json` khóa lại toàn bộ file đã tải. Đây vẫn chỉ là một phần chương trình: ASR Candidate A, challenger training, blind v2 và deployment benchmark còn ở phía sau.

Artifact ASR rerun canonical cũng đã tải và xác minh local: archive `asr-candidate-rerun-af8e0e7.tar.gz` có 786.454 byte, SHA-256 `f2a2cf05b4473e88311746283f598e2d0261f2dab147c554f3b41c860b61fec6`, gồm 11 member/6.170.110 byte nội dung. Sidecar, manifest từng member, prediction provenance và gate đều khớp; `candidate_gate.json` có SHA-256 `72a3d460cf4e49609d30b8150d7e76d102c5b63e25e7e3f83df4a46d8e8d8637`. Bản kê tải local nằm ở `test/program-20261005-014500/asr_candidate_rerun_download_manifest.json`.

Artifact M2M100 EN→VI r8 đã được kéo về local ngay sau khi hoàn tất. Bốn checkpoint 100/200/300/400 đều qua `scripts/verify_checkpoint_download.py`; eval loss giảm tuần tự 2,029509 → 2,007828 → 1,992845 → 1,989010. Checkpoint 400 có 15.341.148 byte, SHA-256 `6c4ab030942831563527a2f77394a0cde9533d4992b358fda68f221541d2f879`, 13 member/20.483.074 byte và index cuối SHA-256 `f2bb1e6fd68a04da95e69d64b59e1d8624f3bb41632a4636da4369cfa752045e`. Bundle toàn round `01-pilot.tar.gz` có 37.381.879 byte, SHA-256 `d4f21609f0e7fddb60e9fa05b457e43636f7a454c21334f3e65c1f20907b1aa1`, 40 member/52.735.651 byte; archive, sidecar, manifest và từng member đều đã được đọc lại và xác minh. Report selection canonical `pilot_r8_lr1e4.tar.gz` có 84.441 byte, SHA-256 `f71c16cfdc1a0e8cae653c1336ab1226b5ff3e9eb273bdb10888c1e2dd0bb60a`, gồm 3 member/341.516 byte và khóa cả predictions/provenance.

Profile M2M100 EN→VI r16/lr5e-5 đã hoàn tất 400 bước; cả bốn checkpoint 100/200/300/400 được tải, đọc lại và xác minh local ngày 06/10/2026. Checkpoint 400 có 28.268.829 byte, SHA-256 `037db3a086c15d2a92da07cd0d8d7b876e4f26b646ebae29f8c901c6cf693e30`; content manifest có 13 member/34.639.045 byte, SHA-256 `0f843f61fde23da6b5d6a9362eec1cee6c7a0e8ed33b3150c441cc99d2cbefa5`; checkpoint index cuối SHA-256 `75fcebf532a54c35e16311e79d2597e836fdec02d0782d7cc14ce8e89b82f688`. `eval_loss` giảm tuần tự 2,041734 → 2,018978 → 2,005621 → 2,001269. Bundle toàn round `01-pilot.tar.gz` có 67.590.627 byte, SHA-256 `4f1999d0a412aae02dfceb25f65321f6a50b9bcc578d72b1237fef20c85349e2`, gồm 40 member/85.725.257 byte; archive, sidecar, manifest và từng member đều đã xác minh. Report selection canonical `pilot_r16_lr5e5.tar.gz` có 86.652 byte, SHA-256 `589d9e4e53fb48275aed602c2f4bb9e3c2c8614521f8595640b1017f581a95b3`, gồm 3 member/348.455 byte. Report ghi 19 lỗi số, 5 lỗi đơn vị, 5 lỗi quantity, 54 lỗi phủ định và 3 lỗi identifier; clinical gate fail nên không promotion.

Profile M2M100 EN→VI r32/lr2e-5 đã hoàn tất 400 bước; cả bốn checkpoint được tải và xác minh local. Checkpoint 400 có 54.144.468 byte, SHA-256 `8b34080a901130530154d19168b8d4dfb53295e7b50c3f910fe0239868ab89de`; content manifest có 13 member/62.950.748 byte, SHA-256 `89902b86a4c95f482c91b2126d3ee8db38ebe04c74547f9c49ce646d79fa539a`; checkpoint index cuối SHA-256 `50eab13daca0177aa999c83a11527ffd41e4bbc4e37767d6063e27393249fe35`. `eval_loss` giảm tuần tự 2,075872 → 2,038227 → 2,027283 → 2,022817. Bundle toàn round có 128.037.121 byte, SHA-256 `cc3e57e60ab13c5bc77cb989980b805191012319b2aaf480f133ae38f577a3b7`, gồm 40 member/151.792.079 byte. Report selection canonical có 85.660 byte, SHA-256 `a71e774aac33658ce012d658cafed630905a3a3f96c33f9363ed62652fc74351`, gồm 3 member/343.180 byte. Report ghi 17 lỗi số, 5 lỗi đơn vị, 5 lỗi quantity, 56 lỗi phủ định và 2 lỗi identifier; clinical gate fail nên không promotion.

Report M2M100 VI→EN zero-shot canonical đã tải và xác minh: archive 90.340 byte, SHA-256 `19c0c78997401d7e9d62ebff36d954cfee9e28bb2a255ed3896ec96a62c189b8`, gồm 3 member/342.426 byte. Report ghi 15 lỗi số, 5 lỗi đơn vị, 5 lỗi quantity và 52 lỗi phủ định; clinical gate fail. Đây là baseline selection của riêng chiều VI→EN, không dùng kết quả EN→VI để thay thế.

Pilot M2M100 VI→EN r8/lr1e-4 đã hoàn tất 400/400 bước với return code 0. Cả bốn checkpoint 100/200/300/400 đã được tải ngay và xác minh local ngày 06/10/2026; `eval_loss` giảm 1,932391 → 1,910451 → 1,895833 → 1,892160 và checkpoint 400 là best. Archive bước 400 có 15.339.377 byte, SHA-256 `557922cbb158af8ce11498b6ca967d9b9e36b4219cd342948fedae9b9db5387c`; content manifest SHA-256 `fd637acc489b90d33833a4c626f900f9a21df31530768921d948c0d06989424a`, gồm 13 member/20.483.083 byte; checkpoint index cuối có SHA-256 `4f0b5b7bfcb8e74ece13489098c28785692f17d00b6f3b0815532aa032b51368`. Bundle toàn round có 37.374.089 byte, SHA-256 `829c4545b9d30d87eee6e739dc9754838c6a67bf512678e98b0b61839a9a512a`, gồm 40 member/52.705.028 byte; archive, sidecar, manifest và từng member đều đã xác minh. Monitor ghi nhận 0 throttle event, peak VRAM process 8.148 MiB và không chạy hai GPU child. Report selection canonical có 92.029 byte, SHA-256 `fc978de99aacf3d736962ce1d42a7d3aeebe67b5b928a64a62ea7031e9be4373`, gồm 3 member/352.545 byte. Report ghi 20 lỗi số, 51 lỗi phủ định, 6 lỗi đơn vị, 6 lỗi quantity và 2 lỗi identifier; clinical gate fail nên không promotion.

Pilot M2M100 VI→EN r16/lr5e-5 đã hoàn tất 400/400 bước với return code 0. Bốn checkpoint đã tải và xác minh local; `eval_loss` giảm 1,943672 → 1,920210 → 1,908257 → 1,903835. Checkpoint 400 có 28.263.835 byte, SHA-256 `c60fe9fe12daddff88bfa02635d66ff241d594cec96f335d8b8116b3d95f92ea`; content manifest SHA-256 `d58e0eeaff00ba86251c734810fd7d2d2d1c31136bedce2b44cc4fd13cf76793`, gồm 13 member/34.639.046 byte; checkpoint index cuối SHA-256 `ac429fa2e7f7513aa7e51d9d869f1586f72b90984a8c88b8897834e3761342af`. Bundle toàn round có 67.585.559 byte, SHA-256 `e3fd481a783aec64600ec1ae0da9eb44dd76bfb4d6ca56f8195ada1ae372924e`, gồm 40 member/85.762.534 byte. Monitor ghi nhận 0 throttle event, peak VRAM process 8.168 MiB. Report selection canonical có 88.021 byte, SHA-256 `d13a72783fd1d6ce6f106c88d79353ae5e424bedf20b62d15d3977c5dbd2dbf3`, gồm 3 member/333.843 byte. Report ghi 18 lỗi số, 50 lỗi phủ định, 4 lỗi đơn vị, 4 lỗi quantity và 1 lỗi identifier; clinical gate fail nên không promotion.

Pilot M2M100 VI→EN r32/lr2e-5 không còn chạy. Lần đầu kết thúc bất thường gần bước 110/400 như nêu trên. Checkpoint 100 đã tải và xác minh local: archive 54.261.566 byte, SHA-256 `003f2f4994d668bb8d88227bf1c2785a20c1356f699eb34b0f354916856b1d43`; content manifest SHA-256 `ce8af31959e5d8ccafd2ced006b9a336927f6da0d734e39aea0e892426a77b71`, gồm 13 member/62.948.179 byte; checkpoint index SHA-256 `ee6f13baa13ce1c730d0b5c3bf5084f359c9278dc10bec14722fc5c1de8824d7`. Tất cả hash và trainer state khớp; `eval_loss=1,973212`. Đây là checkpoint resume duy nhất đã xác minh, chưa phải kết quả selection/safety và không được tải lại.

Recovery waiter PID 375401 đã chờ capacity an toàn rồi `exec` runner lúc 03:20:45Z ngày 08/10/2026. Runner dừng trước GPU training tại completed stage EN→VI r8 vì validator cũ resolve `../.venv-onevoice/bin/python` và `./.venv-onevoice/bin/python` theo working directory của process audit, dù cả hai chỉ khác alias và cùng trỏ tới `/usr/bin/python3.11`; state/checkpoint không hỏng. Commit `da9cfa4` sửa resolution theo stage root thực tế và commit `e3177c2` thêm `--resume-audit` chỉ đọc để kiểm toàn bộ completed stage trước recovery. Remote chưa được pull hoặc resume vì phiên giám sát hiện chỉ được phép read-only; recovery kế tiếp phải audit pass dưới đúng interpreter rồi mới khởi động đúng một runner.

Checkpoint ASR full-data bước 500, 750 và 1.000 đã được watcher phát hành, tải về local và xác minh ngày 06/10/2026. Archive bước 1.000 có 28.880.979 byte, SHA-256 `95596f8bacabf84ccc2b39541758d4db183f077c3bc578b6c8e86b845193898a`; sidecar SHA-256 `2dfd433e44af336d4e0ae30dcd12199e1532fe5fe76dd88fc51cd64d2a1acf9c`, content manifest SHA-256 `79dfc567bd8ac0b35b9a973810e39ae70006994ddc07cc7e43e2a5d787f5eb5d` và checkpoint index SHA-256 `0d17eb104781c68818341a76c91bf672b29e520e549ef14beac3af0cde87e44c` đều khớp. Cả 15/15 regular member (34.775.467 byte nội dung) đều đúng hash, không có path traversal, duplicate, symlink hay special member; `trainer_state.global_step=1000` và `max_steps=2871`. Báo cáo xác minh local có SHA-256 `b72a1e6b596a0acf9b866ef398c038b12a43c00c94094a7b228179edfd4fe70b`.

`eval_loss` tiếp tục giảm từ 1,5151 ở bước 250 qua 1,1474 ở bước 500, 0,9435 ở bước 750 xuống 0,7191 ở bước 1.000. `eval_wer` lần lượt là 67,3243; 39,1998; 39,9387 và 33,7840, nên checkpoint 1.000 trở thành best mới theo policy `eval_wer` lower-is-better. Đây vẫn chỉ là validation trend, không phải locked-test safety evidence và chưa cho phép promotion.

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

License metadata được pin cùng revision trong `configs/model_bakeoff.yaml`. Đây là gate kỹ thuật fail-closed, không thay thế tư vấn pháp lý. VinAI chỉ được mở bằng `--approve-research-license <candidate-id>`; cờ này không bao giờ đổi `production_eligible` thành true. Danh sách approval và quyết định license chuẩn của toàn bộ candidate được khóa vào selection snapshot/identity. Blind runner tự tính lại quyết định từ config, từ chối approval trùng, sai hoặc candidate lạ, và không chấp nhận winner nếu snapshot thiếu đúng explicit approval đã dùng trong bake-off; scope production vẫn bắt buộc `production_eligible=true`.

## Dữ liệu công bằng

- Train: giữ nguyên manifest train đã audit.
- Model selection: `data/eval/mt_selection_dev.jsonl` (256 cặp) và `data/eval/asr_selection_dev.jsonl` (384 câu), chỉ sinh từ validation. Bake-off preflight tự tính lại fingerprint từ payload, khóa uniqueness và kiểm tra ID/cặp dịch/transcript/audio hash/speaker/recording group không giao cả train lẫn mọi locked test hay safety manifest trước khi dùng GPU.
- Terminology challenge set: `data/eval/terminology_challenge_set.jsonl` khóa 128 lỗi validation có provenance (54 EN→VI, 56 VI→EN, 18 ASR). Builder chỉ nhận prediction khớp chính xác ID/reference/source trong manifest validation; archive MT còn phải khớp sidecar provenance và manifest SHA-256. Tất cả mục vẫn `pending`, evaluation-only và không đủ điều kiện auto-correction cho tới human review.
- Blind locked test v2: `data/eval/blind_test_v2.lock.json` hiện là `awaiting_unseen_data`. Không tự chế blind set từ dữ liệu model đã nhìn thấy. Schema v2 yêu cầu tối thiểu **256 cặp MT** và **384 clip ASR** thật sự unseen; các hàng có thể thuộc nhiều lát nhưng không được nhân bản để tăng đếm.
- Test/safety cũ không được mở lặp lại để chọn model.

Selection dev chứa lát thuốc, liều, số, đơn vị, phủ định, thuật ngữ và code-switch. ASR còn giữ accent/role/recording-condition để báo riêng. Blind v2 chỉ được khóa khi mỗi cặp MT dùng được cho cả hai chiều và đạt 256 mẫu/chiều. MT phải dùng schema chuẩn `source_language=en`, `target_language=vi`; ASR phải khai báo ngôn ngữ Việt, kể cả `vi-code-switch`, cùng `speaker` và `group` ổn định để kiểm tra speaker/recording-disjoint. Mỗi lát thuốc, liều, số, đơn vị, phủ định, thuật ngữ và code-switch phải có ít nhất 32 mẫu cho từng task. ASR còn bắt buộc ít nhất 64 clip Bắc, 64 Trung, 64 Nam, 64 bác sĩ, 64 bệnh nhân, 96 clip nhiễu, **32 speaker độc lập và 32 recording group độc lập**; các thuộc tính có thể giao nhau trên cùng clip. Script lưu lại thống kê coverage trong lock, tính lại khi verify và fail-closed nếu thiếu quota, thiếu ID/fingerprint, có ID/pair/audio trùng trong chính blind set, hoặc có ID, text fingerprint, pair fingerprint, audio SHA, speaker hay recording group giao train/selection/test cũ. Audit không tin rằng các tập lịch sử đã có sẵn fingerprint: nó tự tính lại pair/text fingerprint từ payload và tự băm file audio khi SHA bị thiếu, nên safety MT/ASR cũ không thể bị né bằng cách đổi ID. Prediction ASR giữ `speaker/group`; WER 95% CI bootstrap theo recording group thay vì coi các câu cùng buổi ghi là độc lập. Report sau inference phải khớp chính xác số hàng, số mẫu từng lát, số speaker/group và số cluster trong lock. Metadata khai báo sai, file audio thiếu hoặc bị sửa đều bị từ chối.

Blind v2 không chỉ kiểm tra safety. Nó dùng ngưỡng release đã khóa SHA-256 trong `configs/model_bakeoff.yaml` và lấy nội dung từ `configs/accuracy_program.yaml`: cận trên 95% CI của ASR WER phải không vượt ngưỡng, còn cận dưới 95% CI của MT BLEU và chrF2 phải đạt ngưỡng cho từng chiều. Preflight và blind runner đều từ chối nếu checksum policy thay đổi. CER và code-switch WER cũng phải hữu hạn, code-switch WER phải đạt policy. Thiếu metric/CI, NaN hoặc point estimate/CI không đạt đều khóa promotion.

Mỗi slot blind chỉ được mở cho winner đúng task/chiều trong snapshot `selection_comparison-<identity>.json` bất biến; tên snapshot lấy từ digest của winner bindings nên các vòng bake-off sau không ghi đè vòng cũ. `comparison.json` có thể tiếp tục nhận blind/deployment result mà không làm hỏng checksum selection đã khóa. Trước khi suy luận, runner khóa SHA-256 của snapshot này, manifest cây adapter và blind manifest vào specification; adapter bị sửa tại cùng đường dẫn cũng bị từ chối. Mỗi slot giữ mutex cấp OS suốt phiên đánh giá để hai lệnh đồng thời không thể spawn inference trùng; cập nhật lock dùng mutex state riêng, đọc lại checksum và ghi nguyên tử nên các slot chạy song song không ghi đè kết quả nhau. Mutex tự nhả nếu process chết, còn slot đã mở chỉ được resume với đúng candidate digest. Report có provenance sidecar chứa checksum/bytes và toàn bộ binding này, nên resume chỉ được dùng lại report đã xác minh; report có sẵn nhưng thiếu provenance hoặc bị sửa sẽ fail-closed thay vì được chấm lại.

## Successive halving và winner gate

Mỗi GPU child chạy tuần tự, 35% VRAM/process, hard memory 40%, rolling utilization 70%, resume 55% và hard reaction 74% — luôn dưới 75%. Trước child train/eval đầu tiên, runner không dùng GPU và chờ đến khi utilization không vượt resume threshold đồng thời VRAM trống đủ chứa toàn bộ hard budget 40%. Khi resume có ít nhất 50 mẫu lịch sử được quy đúng process-group PID hoặc Linux `NSpid`, headroom tự điều chỉnh thành peak process lớn nhất cộng phần lớn hơn giữa 25% và 2 GiB, vẫn bị chặn trên bởi hard budget 40%; launch ngắn và telemetry post-spawn không được dùng để hạ mức bảo vệ. Log `pre_spawn_capacity.jsonl` ghi bằng chứng mỗi phút và PID sở hữu stage ngăn launch trùng khi đang chờ. Sau khi spawn, stage lấy mẫu tài nguyên mỗi 1 giây; preflight từ chối ngưỡng không hữu hạn, sai thứ tự hoặc vượt trần trước khi khởi chạy child.

Nếu training bị gián đoạn, runner chỉ resume từ checkpoint đã đối chiếu đủ index/archive/sidecar/member manifest/Trainer state/live tree, ghi log và lịch sử riêng cho từng attempt, và giới hạn tối đa ba attempt. Không có checkpoint hợp lệ hoặc hết budget retry sẽ dừng fail-closed; không tự train lại từ đầu và không biến lỗi hạ tầng thành bằng chứng benchmark.

Monitor VRAM đối chiếu cả PID trong container và chuỗi `NSpid` Linux để tìm đúng host PID mà `nvidia-smi` báo. Chỉ khi không có ánh xạ namespace mới dùng chênh lệch snapshot trước/sau spawn; nhờ vậy job GPU khác khởi động đồng thời không bị cộng nhầm vào bộ nhớ của child đang train và không tạo hard-stop giả.

Selection comparison phải chứa checksum của policy đa metric/95% CI và toàn bộ input selection-dev + accuracy policy. Blind runner từ chối comparison cũ, thiếu policy hoặc lệch checksum. Mỗi lệnh selection-dev còn mang SHA-256 của `run_model_bakeoff.py` đã nạp lúc waiter khởi động; child benchmark đối chiếu nó với runner hiện hành trên đĩa **trước khi nạp model hoặc CUDA**. Vì vậy waiter sống lâu từ commit cũ sẽ dừng fail-closed thay vì tạo inference bằng logic cũ. Chỉ sau khi PID cũ đã kết thúc và state xác nhận không có GPU child, runner hiện hành mới được resume đúng state directory; artifact hợp lệ có thể được tái dùng, còn stage thiếu khóa thế hệ phải chạy lại và phát hành immutable selection snapshot mới trước khi blind test được phép mở.

Guard thế hệ cũng chạy ở đầu mọi `run_stage`, nên process cha stale không thể lách qua bằng cách launch training, blind hay deployment child sau một benchmark đã hoàn tất. Việc thay code khi bake-off đang sống buộc dừng an toàn ở ranh giới stage kế tiếp.

Trước bất kỳ GPU stage bake-off nào, runner bây giờ kiểm tra lại locked evaluation của cả hai Candidate A. `status` chỉ được là `pass` hoặc `fail` với `error=null`; `error` hạ tầng, gate rỗng, sai adapter, sai base model/revision hay sai promotion flag đều chặn bake-off trước khi nạp CUDA. Gate phải có đủ aggregate metric, toàn bộ critical/policy slice, trạng thái nhất quán với từng check, locked-input hash hợp lệ và cả hai resource run trả mã 0 dưới đúng policy GPU. ASR bắt buộc có archive, sidecar và canonical content manifest khớp từng member. MT legacy không có manifest chỉ được miễn theo cấu hình `legacy_reference_only=true`, vì gate `fail` đã biết và model này không thể tự promotion. Hash gate/archive của hai task được ghi vào state, comparison và selection identity; thay bằng chứng sau khi chọn winner sẽ làm snapshot blind không còn khớp.

Resume không tin file `candidate_a_freeze.json` chỉ vì nó đã tồn tại. Runner băm lại từng file của cả adapter MT/ASR, so khớp root/path/bytes/SHA-256/manifest digest và kiểm tra adapter binding trong program state; freeze cũ, bị sửa hoặc trỏ sang checkpoint khác sẽ dừng fail-closed trước bake-off.

State resume còn khóa đúng program-state path, SHA-256 của config, `research`/`production` scope và danh sách phê duyệt research-license. State legacy chưa có binding chỉ được nâng cấp khi chưa có stage nào complete; nếu đã có GPU result thì runner từ chối thay vì suy diễn nó thuộc invocation hiện tại.

Mỗi stage ghi PID và process-group ngay sau `Popen`. Nếu parent gián đoạn, resume kiểm tra PID cũ trước khi ghi đè state; state legacy chưa có PID được đối chiếu exact command qua `/proc/*/cmdline`. Chỉ khi không còn process khớp mới được launch lại, ngăn hai GPU child chạy song song sau crash.

1. Zero-shot trên cùng selection dev.
2. Pilot 400 steps, effective batch 32. Mỗi challenger/chiều nhận cùng ba profile LoRA: r8/lr1e-4, r16/lr5e-5 và r32/lr2e-5; alpha lần lượt 16/32/64. Chỉ profile qua safety tốt nhất của từng candidate/chiều đi tiếp.
3. Loại candidate fail safety hoặc thua rõ; giữ tối đa hai.
4. Semifinal 2.000 steps từ cùng base revision.
5. Full train giữ các candidate không bị candidate dẫn đầu Pareto-dominance theo 95% CI; một metric tốt không được che một metric khác kém rõ rệt.
6. MT EN→VI và VI→EN được chọn độc lập.

Giới hạn `keep` của successive halving được áp dụng riêng trong từng chiều MT, không xếp chung hai chiều. Vì vậy candidate EN→VI điểm cao không thể chiếm hết slot và loại toàn bộ VI→EN; finalist Pareto/95% CI cũng được xét trong từng chiều. ASR không có phân chiều nên vẫn dùng một ranking chung.

Hyperparameter không được chọn theo locked test. Toàn bộ profile dùng cùng selection-dev checksum, số bước và sample cap; profile thắng được khóa trong runtime config, stage name và result record rồi dùng lại cho semifinal/full. Sửa profile làm đổi config SHA-256 và bị invocation binding từ chối khi resume.

So sánh theo effective train samples, không theo thời gian. Safety là hard gate: thuốc/liều/số/đơn vị/phủ định phải 0 failure; terminology và code-switch cũng có policy riêng. Sau safety, MT bắt buộc đủ cả BLEU và chrF2 theo chiều cùng bootstrap 95% CI; ASR bắt buộc đủ WER, CER và code-switch WER, trong đó WER có bootstrap 95% CI. Chỉ thay Candidate A khi ít nhất một metric có CI tốt hơn rõ rệt và không metric có CI nào thụt lùi rõ rệt; CER/code-switch vẫn phải hữu hạn và code-switch phải qua policy riêng. Winner search duyệt toàn bộ challenger đã qua safety theo thứ hạng và chọn challenger cao nhất thực sự thỏa luật đa metric; một model đứng đầu point estimate nhưng có CI regression không được che mất model xếp sau có Pareto improvement hợp lệ.

Candidate A MT cũng được benchmark độc lập theo từng chiều; lỗi an toàn hoặc khoảng tin cậy của EN→VI không được làm thay đổi quyết định VI→EN và ngược lại. Metric hay bootstrap 95% CI thiếu, không hữu hạn hoặc đảo cận đều bị loại fail-closed trước ranking.

## Chứng cứ cũ đã sửa

`data/reports/experiments/decoding/mt_val_greedy_rescored.json` dùng prediction của NLLB greedy. File đã được gắn `model=facebook/nllb-200-distilled-600M` và `evidence_status=invalid_for_model_comparison`; không còn là bằng chứng VinAI Translate. NLLB greedy/beam gốc vẫn hợp lệ vì có prediction, model ID và revision rõ ràng.

## Lệnh vận hành

```bash
python scripts/build_selection_dev.py
python scripts/build_terminology_challenge_set.py
python scripts/run_model_bakeoff.py --preflight

# Trước mọi recovery, audit read-only completed stages dưới đúng interpreter.
# Lệnh trả mã 2 nếu command/evidence không còn tương thích.
python scripts/run_model_bakeoff.py --resume-audit \
  --state-dir gpu-runs/model-bakeoff \
  --python ./.venv-onevoice/bin/python

# Chỉ chạy sau khi current program state complete. Process này có thể chờ mà không dùng GPU.
python scripts/run_model_bakeoff.py \
  --execute --wait-current \
  --current-program-state gpu-runs/program-YYYYMMDD-HHMMSS/program_state.json \
  --asr-candidate-output gpu-runs/program-YYYYMMDD-HHMMSS/asr-candidate-rerun-<commit> \
  --state-dir gpu-runs/model-bakeoff

# Khi có blind v2 thật sự chưa từng bị model nhìn thấy:
python scripts/run_blind_candidate_suite.py --action lock \
  --mt-manifest /secure/blind_v2_mt.jsonl \
  --asr-manifest /secure/blind_v2_asr.jsonl
```

Runner có resume theo command digest, checkpoint watcher, archive/checksum từng round và từ chối chạy hai command khác nhau dưới cùng stage. `--resume-audit` là đường chỉ đọc: nó đối chiếu digest lệnh đã lưu, interpreter alias theo đúng stage root, runner generation và output evidence của mọi stage `complete` mà không sửa state hoặc spawn subprocess; phải pass trước khi tạo recovery waiter. Mỗi file đầu ra của stage còn được khóa path/bytes/SHA-256 trong state; file bị thay thế sau khi hoàn tất sẽ làm resume fail-closed. Benchmark do waiter cũ hoàn tất nhưng chưa có khóa output sẽ tự chạy lại đúng đường CPU scoring từ prediction checkpoint đã xác minh bằng `--resume-scoring`, không nạp model hoặc suy luận GPU lần nữa. Trước khi bất kỳ adapter bake-off nào được benchmark, runner đọc lại tar, sidecar và content manifest của mọi round rồi đối chiếu toàn bộ cây adapter sống theo path/bytes/SHA-256; chỉ có `adapter_config.json` là không đủ. Manifest freeze/winner dùng đường dẫn POSIX ổn định giữa local và GPU, đồng thời từ chối symlink ở root hoặc bất kỳ file con nào. Khi blind v2 chưa sẵn sàng, runner dừng ở trạng thái chờ thay vì tự mở test cũ.

## Deployment reference

Dự án giữ immutable evidence QCS6490 đã đo ngày 26/09/2026. Tuy nhiên trang Qualcomm hiện tại được kiểm tra lại ngày 05/10/2026 không còn liệt kê QCS6490 cho Whisper-Small-Quantized. Vì vậy evidence cũ không bị xóa, nhưng promotion mới bắt buộc rerun trên board QCS6490 và báo latency p50/p95, power trung bình/p95, nhiệt độ peak, peak RAM/VRAM và model bytes. Không suy diễn khả năng deployment của winner fine-tuned chỉ từ reference OpenAI Whisper-small.

`deployment_selected_winners.json` phải là schema version 1, ghi `measurement_source=physical_board`, timestamp có timezone và metadata board/chipset/OS. Metadata tự khai không đủ: phải chạy `scripts/capture_qcs6490_identity.py` ngay trên board, giữ JSON device-tree/sysfs bất biến dưới `data/reports/model_bakeoff/board-evidence/`, và để finalizer khóa SHA-256. Gate yêu cầu ARM64 cùng cả model board và chuỗi `qcom` compatible nhận dạng QCS6490/RB3 Gen 2; timestamp identity phải nằm trong 24 giờ của phiên benchmark. Arduino, cloud host, evidence cũ hoặc file ngoài thư mục evidence đều bị từ chối. Mỗi winner phải khớp task, chiều, candidate ID và SHA-256 manifest của đúng adapter; artifact triển khai phải nằm dưới `models/`, có path/SHA-256/byte size khớp file thật. Mỗi winner còn bắt buộc parity evidence schema 2 bất biến dưới `board-evidence/parity/`: sealer đọc float/compiled prediction JSONL trên đúng cùng manifest, bắt buộc provenance của float checkpoint và compiled capture, đối chiếu decoding cùng toàn bộ metadata ngoài hypothesis, tự chấm metric, paired-bootstrap 1.000 lần và khóa SHA-256 của manifest, hai prediction files, hai provenance files cùng compiled artifact. Point estimate và cận trên 95% CI của relative degradation phải ≤2%; ASR cần ít nhất 32 group/speaker độc lập; float lẫn compiled output phải zero failure trên drug/dose/number/unit/negation/terminology/code-switch. Raw measurements không được tin từ draft: mỗi winner phải trỏ tới một JSON bất biến dưới `board-evidence/measurements/`, khóa version/capture source/timestamp, identity SHA, artifact SHA/bytes, candidate/adapter binding, latency/power/thermal samples, nguồn sensor và peak RAM/VRAM. Finalizer khóa SHA-256 parity/measurement artifacts rồi mới copy dữ liệu sang report. Report phải chứa ít nhất 30 mẫu riêng cho latency, power và thermal. Runner đọc lại evidence, tự tính latency p50/p95, power trung bình/p95 và nhiệt độ peak; artifact bị sửa, parity/measurement/provenance binding sai, decoding khác nhau, metric không khớp mẫu, sensor thiếu, mẫu không hữu hạn/phi vật lý hoặc sample count thiếu đều fail-closed. VRAM được phép bằng 0 nếu chạy NPU/CPU. Report cloud cũ, report của model khác hoặc số đo giả/thiếu đều bị reject.

Không soạn report cuối bằng tay. Khi blind v2 pass, state machine tự tạo draft đã khóa đúng ba winner, checksum cây adapter và checksum `comparison.json`; lệnh tương đương để phục hồi thủ công là:

```bash
python scripts/prepare_deployment_benchmark.py --action template \
  --selection-comparison data/reports/model_bakeoff/comparison.json
```

CLI chỉ tạo draft sau khi `comparison.json` đã ở trạng thái `blind_complete`. Trên QCS6490 thật, trước tiên chạy `python scripts/capture_qcs6490_identity.py --output data/reports/model_bakeoff/board-evidence/qcs6490-identity.json`. Dùng sampler chính thức để chạy trực tiếp argv inference (không qua shell), bỏ warm-up, lấy ít nhất 30 latency và poll power/thermal sysfs trong suốt mỗi run; scale phải khai báo theo đơn vị thật của sensor. `--minimum-duration-seconds 1800` buộc soak tối thiểu 30 phút để lộ throttling, còn mỗi invocation vẫn có timeout hữu hạn:

```bash
python scripts/capture_qcs6490_runtime.py \
  --power-sensor-path /sys/<board-power-sensor> --power-scale-to-mw <scale> \
  --temperature-sensor-path /sys/class/thermal/<zone>/temp --temperature-scale-to-c 0.001 \
  --runs 30 --warmup-runs 3 --sample-interval-ms 100 \
  --timeout-seconds 300 --minimum-duration-seconds 1800 \
  --output <raw-sampler-output.json> \
  -- /opt/onevoice/bin/infer-winner --input <locked-fixture>
```

Sampler xác minh device-tree/sysfs QCS6490 trước khi spawn command, không ghi stdout/stderr lâm sàng, dừng cả process group khi timeout/lỗi sensor, và fail nếu không quan sát được RSS dương. Power/thermal giữ toàn bộ mẫu poll; latency giữ một mẫu cho mỗi invocation; `peak_vram_bytes=0` cho đường NPU/CPU không có VRAM. Output schema 1 có `capture_source=qcs6490_runtime_sampler`, timestamp có timezone, nguồn sensor và peak RAM. Không tự đổi raw JSON thành evidence: dùng sealer để kiểm lại live board, timestamp, sample contract và artifact binding rồi ghi bất biến:

Trước khi finalize, chạy lại cùng manifest khóa qua winner float và compiled artifact rồi seal parity. Float predictions phải giữ file `.provenance.json` do baseline runner tạo. Compiled predictions phải đi qua capture wrapper; ba placeholder là token argv độc lập, mỗi token xuất hiện đúng một lần, và command phải tự ghi JSONL vào `{output_predictions}`. Wrapper không dùng shell, không ghi stdout/stderr lâm sàng, timeout hữu hạn, từ chối overwrite và xóa output chưa hoàn tất khi runtime lỗi:

```bash
python scripts/capture_compiled_predictions.py \
  --task mt --direction en_to_vi \
  --candidate-id <winner-id> \
  --adapter-manifest-sha256 <64-hex> \
  --runtime-kind qnn_context_binary \
  --artifact models/<winner-artifact> \
  --manifest <locked-parity-manifest.jsonl> \
  --output-predictions <compiled-predictions.jsonl> \
  --output-provenance <compiled-predictions.provenance.json> \
  --num-beams <same-as-float> --timeout-seconds 3600 \
  -- /opt/onevoice/bin/infer-compiled \
  --artifact '{artifact}' --manifest '{manifest}' \
  --output '{output_predictions}'
```

MT bắt buộc chỉ định chiều; ASR bỏ `--direction` và prediction metadata phải giữ group/speaker của blind set:

```bash
python scripts/seal_quantization_parity.py \
  --task mt --direction en_to_vi \
  --candidate-id <winner-id> \
  --adapter-manifest-sha256 <64-hex> \
  --artifact models/<winner-artifact> \
  --manifest <locked-parity-manifest.jsonl> \
  --reference-predictions <float-predictions.jsonl> \
  --reference-provenance <float-predictions.jsonl.provenance.json> \
  --quantized-predictions <compiled-predictions.jsonl> \
  --quantized-provenance <compiled-predictions.provenance.json> \
  --output data/reports/model_bakeoff/board-evidence/parity/<winner>.json
```

Không chỉnh tay parity report: sealer tự chấm SacreBLEU/chrF2 cho MT hoặc WER/CER/code-switch WER cho ASR, tính paired-bootstrap CI và từ chối mọi clinical slice có failure.

```bash
python scripts/seal_qcs6490_measurement.py \
  --identity-evidence data/reports/model_bakeoff/board-evidence/qcs6490-identity.json \
  --artifact models/<winner-artifact> \
  --raw-measurement <raw-sampler-output.json> \
  --task mt --direction en_to_vi \
  --candidate-id <winner-id> \
  --adapter-manifest-sha256 <64-hex> \
  --output data/reports/model_bakeoff/board-evidence/measurements/<winner>.json
```

Lặp lại cho hai chiều MT và ASR (ASR bỏ `--direction`), rồi chép identity/parity/measurement evidence cùng artifact thực tế về đúng project. Trong `.draft.json` chỉ điền `measured_at`, `device.os`, `device.identity_evidence_path`, `artifact_path`, `parity_evidence_path` và `measurement_evidence_path`; không tự điền metric tổng hợp. Chạy `prepare_deployment_benchmark.py --action finalize` với cùng comparison. CLI tự khóa checksum evidence, tính các percentile/peak, SHA-256/kích thước từng artifact và chỉ tạo `deployment_selected_winners.json` nếu toàn bộ quantization-parity và physical-board gate pass; draft legacy thiếu parity/measurement binding bị từ chối và draft/report có sẵn không bị ghi đè.
