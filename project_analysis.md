# OneVoice / MediVoice Edge — Project Analysis & Execution Ledger

> Cập nhật gần nhất: **08/10/2026 (Asia/Bangkok)**
> Chủ dự án: **1 người**
> Cách dùng máy: kiểm tra dữ liệu và chạy thử trên máy local; **chỉ thuê GPU online khi fine-tune**
> Trạng thái tổng: **DỮ LIỆU/GPU-READY, CANDIDATE A ĐÃ HUẤN LUYỆN XONG NHƯNG FAIL CLINICAL GATE, BAKE-OFF CHƯA HOÀN TẤT, RELEASE GATE ĐANG KHÓA** — NLLB-600M và PhoWhisper-small Candidate A chỉ là reference. Các profile M2M100 đã hoàn tất đều fail clinical gate; VI→EN r32 bị gián đoạn sau checkpoint 100 đã xác minh và recovery chưa được chạy lại trên remote. Whisper-small multilingual/PhoWhisper-base, blind v2, INT8/QNN parity và phép đo QCS6490 vật lý còn thiếu.
> Nguồn sự thật vận hành: **chỉ `project_analysis.md`**. Mọi thay đổi trạng thái, giả định, kết quả đo và blocker phải cập nhật ngay dưới phần liên quan trong file này; hai DOCX proposal là bản lưu lịch sử và phải giữ nguyên byte-for-byte.

## Cách đọc nhanh và mục tiêu của từng thuật ngữ

Từ phần này trở xuống, thuật ngữ kỹ thuật luôn đi kèm nghĩa ngắn hoặc luồng đầu vào → đầu ra. Nếu một thuật ngữ xuất hiện lại, nghĩa và mục tiêu vẫn giữ như bảng dưới.

| Thuật ngữ | Nghĩa đơn giản / đầu vào → đầu ra | Mục tiêu | Lý do cần có |
|---|---|---|---|
| ASR | Nhận dạng giọng nói: **voice VN → text VN** hoặc **voice EN → text EN** | Chép đúng lời người nói thành chữ cùng ngôn ngữ. | MT chỉ dịch tốt khi câu chữ đầu vào đúng. |
| MT | Dịch máy: **text VN ↔ text EN** | Dịch hai chiều nhưng giữ nguyên số, đơn vị, tên thuốc và ý phủ định. | Đây là phần chuyển ngôn ngữ chính của OneVoice. |
| TTS | Tổng hợp giọng nói: **text VN → voice VN** hoặc **text EN → voice EN** | Đọc bản dịch thành tiếng rõ ràng. | Người dùng có thể nghe thay vì chỉ nhìn màn hình. |
| VAD | Phát hiện đoạn có tiếng nói: **audio liên tục → đoạn có lời** | Bỏ khoảng lặng và xác định lúc người dùng nói xong. | Giảm xử lý thừa và giúp phản hồi nhanh hơn. |
| EDA | Kiểm tra dữ liệu trước khi dùng | Tìm dữ liệu rác, lệch, trùng, thiếu hoặc sai nhãn. | Nếu dữ liệu đầu vào sai thì model sẽ học sai. |
| QC | Kiểm tra chất lượng file âm thanh | Giữ file nghe được; loại file im lặng, hỏng hoặc bất thường thật sự. | Không xóa máy móc chỉ vì clip ngắn. |
| Merge | Gộp các nguồn đã kiểm tra | Tạo một bộ train/validation/test sạch và có quy tắc. | Tránh cùng người nói hoặc cùng audio xuất hiện ở nhiều tập. |
| Train | Tập dùng để model học | Cải thiện model trên dữ liệu đúng mục tiêu. | Đây là dữ liệu được phép tác động lên trọng số model. |
| Validation | Tập chọn checkpoint | Chọn phiên bản tốt nhất trong lúc train. | Tránh chọn model bằng test và làm sai kết quả cuối. |
| Test | Tập kiểm tra cuối, được khóa | Đo chất lượng sau khi đã chọn xong model. | Giữ kết quả khách quan. |
| Baseline | Kết quả model gốc trước fine-tune | Tạo mốc để biết fine-tune có thật sự tốt hơn không. | Không thể kết luận tiến bộ nếu thiếu mốc so sánh. |
| Fine-tune | Huấn luyện thêm model có sẵn | Cho model hiểu tốt hơn giọng nói và câu y tế của dự án. | Rẻ và thực tế hơn train model từ đầu. |
| Augmentation | Tạo biến thể âm thanh khi train | Giúp ASR chịu được tốc độ nói, nhiễu, điện thoại và âm lượng khác nhau. | Tăng độ bền mà không làm bẩn validation/test. |
| Leakage | Dữ liệu bị rò giữa train/validation/test | Đưa overlap về 0 theo người nói, nhóm ghi âm và audio trùng bit. | Có leakage sẽ làm điểm số cao giả. |
| PII | Thông tin có thể nhận diện cá nhân | Loại tên bệnh nhân, email, số điện thoại thật. | Bảo vệ riêng tư dữ liệu y tế. |
| WER / CER | Tỷ lệ sai theo từ / ký tự của ASR | Số càng thấp càng tốt. | Đo ASR chép lời sai bao nhiêu. |
| BLEU / chrF | Điểm so khớp bản dịch MT | Số càng cao thường càng tốt, nhưng vẫn phải kiểm tra y khoa. | Điểm dịch chung không tự phát hiện sai số/liều thuốc. |
| RTF | Thời gian xử lý chia cho thời lượng audio | Nhỏ hơn 1 nghĩa là xử lý nhanh hơn thời gian phát audio. | Dùng để xem thiết bị có đáp ứng gần thời gian thực không. |
| Quantization | Giảm độ chính xác số, ví dụ float → INT8 | Làm model nhỏ và chạy nhanh hơn trên thiết bị. | Chỉ chấp nhận nếu chất lượng y khoa không giảm đáng kể. |
| Preflight | Kiểm tra trước khi chạy tốn tiền | Chặn train nếu dữ liệu, hash, model hoặc cấu hình chưa đúng. | Tránh thuê GPU rồi mới phát hiện lỗi. |
| Checkpoint | Bản model được lưu trong lúc train | Cho phép chọn bản tốt nhất và tiếp tục khi bị gián đoạn. | GPU thuê có thể hết phiên hoặc bị dừng. |
| QNN / QCS6490 | Công cụ Qualcomm / chip đích | Đưa model đã kiểm chứng lên thiết bị thật. | Chỉ số trên PC không thay thế được đo trên board. |

### Bốn model chính và việc mỗi model phải làm

| Model | Nhiệm vụ đơn giản | Mục tiêu trong OneVoice | Lý do dùng |
|---|---|---|---|
| `vinai/PhoWhisper-small` | **voice VN → text VN** | ASR chính cho tiếng Việt y tế và câu trộn Việt–Anh. | Được xây từ Whisper đa ngôn ngữ và đã học nhiều giọng Việt. |
| `distil-whisper/distil-small.en` | **voice EN → text EN** | ASR tiếng Anh để chạy baseline và đo trên Eka. | Nhẹ hơn Whisper lớn và phù hợp tiếng Anh; không dùng cho tiếng Việt. |
| `facebook/nllb-200-distilled-600M` | **text VN ↔ text EN** | Một checkpoint MT dịch hai chiều. | Là model dịch chuyên dụng, dễ kiểm soát hơn chatbot. |
| Piper VI/EN | **text VN/EN → voice VN/EN** | Đọc bản dịch thành tiếng ở ngôn ngữ đích. | Nhẹ, chạy local và đã có model giọng Việt/Anh sẵn. |

Một số từ dùng trong log chi tiết:

- `manifest`: danh sách các mẫu được phép dùng và đường dẫn tới dữ liệu; mục tiêu là để mọi lần chạy dùng đúng cùng một tập.
- `hash` hoặc SHA-256: dấu vân tay của file; mục tiêu là phát hiện file thay đổi hoặc audio trùng bit.
- `code-switch`: một câu trộn Việt–Anh; mục tiêu là ASR vẫn chép đúng cả hai phần.
- `ONNX`: định dạng model trung gian; mục tiêu là chuyển model từ môi trường train sang môi trường chạy trên thiết bị.
- `INT8`: dạng số 8-bit; mục tiêu là giảm kích thước và tăng tốc model sau khi đã kiểm tra chất lượng.
- `human QA`: người thật nghe/đọc để xác nhận; mục tiêu là kiểm tra những điều máy không kết luận chắc chắn.

Luồng chính, viết đơn giản:

```text
voice VN → ASR → text VN → MT → text EN → TTS → voice EN
voice EN → ASR → text EN → MT → text VN → TTS → voice VN
```

## 0. Nhật ký thực thi

Đây là log kỹ thuật chi tiết để truy vết. Cách đọc mỗi dòng là: **đã làm gì → vì sao cần làm → kết quả**. Phần giải thích ngắn và mục tiêu của từng bước nằm tại các mục 1–14 bên dưới.

| Thời điểm | Trạng thái | Việc đã làm và lý do |
|---|---|---|
| 09/10/2026 | ✅ Verified QCS6490 measurement sealing | Thêm `seal_qcs6490_measurement.py` để chạy ngay trên board: đọc lại live device-tree/sysfs và bắt buộc khớp identity evidence, kiểm raw sampler JSON bounded/exact-schema, tối thiểu 30 latency/power/thermal samples, sensor, memory, timestamp trong cùng phiên và compiled artifact regular-file. Output bất biến bind task/direction/candidate/adapter, identity SHA, artifact SHA/bytes và raw-input SHA; không overwrite. Evidence tạo ra đã được integration-test trực tiếp với finalizer. Full local suite: 438 pass, 1 skip; compileall và dependency check pass. Chưa có raw capture/board thật nên đây là verified tooling, không phải deployment evidence. |
| 09/10/2026 | ✅ Physical power/thermal deployment gate | QCS6490 deployment draft/report nay bắt buộc một measurement-evidence JSON bất biến cho từng winner thay vì tin raw array chép tay trong draft. Evidence nằm dưới `board-evidence/measurements`, được khóa SHA-256 và bind vào board identity, task/direction/candidate/adapter, compiled artifact cùng timestamp phiên; finalizer chỉ lấy latency/power/thermal/sensor/memory từ artifact đó. Validator đọc lại evidence, tính latency p50/p95, power average/p95 và thermal peak, yêu cầu tối thiểu 30 mẫu mỗi loại, từ chối evidence bị sửa, binding sai, số phi vật lý, metric lệch, sensor thiếu và draft legacy. Power/thermal là metric bắt buộc bằng code, hợp nhất với danh sách cấu hình cũ để giữ nguyên config SHA của bake-off đang chờ recovery. Full local suite: 430 pass, 1 skip; compileall và dependency check pass. Chưa có phép đo board thật nên Phase F vẫn mở. |
| 08/10/2026 | ✅ TTS startup degradation | `allow_text_only_tts_fallback` nay bao phủ cả lỗi load TTS: ASR/MT/cache/audio vẫn phải ready, nhưng riêng TTS thiếu/hỏng có thể đưa pipeline vào `operational_mode=text_only` với mã lỗi ổn định không lộ chi tiết backend. Request an toàn giữ bản dịch và không dispatch synthesis; playback xác nhận vẫn fail-closed khi TTS chưa ready. Tắt fallback giữ hành vi propagate lỗi. Full local suite: 428 pass, 1 skip; compileall pass. |
| 08/10/2026 | ✅ Safe TTS degradation | Pipeline có fallback text-only cấu hình rõ khi TTS artifact/device/output/runtime hỏng sau ASR/MT an toàn: giữ bản dịch, phát mã lỗi ổn định đã khử dữ liệu nhạy cảm, không phát audio lỗi và ghi telemetry riêng. Có thể tắt fallback để giữ fail-closed cũ. Full local suite: 424 pass, 1 skip; compileall và dependency check pass. |
| 08/10/2026 | ✅ Clinical playback confirmation | Audio sink revalidate source/translation safety ngay trước phát; CLI file mode yêu cầu người dùng nhập đúng `PLAY`, còn cache action chỉ phát qua API xác nhận rõ. Không có xác nhận hoặc validation fail thì không phát audio. Full local suite: 416 pass, 1 skip. |
| 08/10/2026 | ✅ Bake-off recovery hardening | Sửa đối chiếu interpreter alias tương đối theo stage root thực tế thay vì working directory của auditor; thêm `--resume-audit` chỉ đọc để kiểm command digest, interpreter identity, runner generation và output evidence của mọi completed stage trước recovery. Remote chưa pull/restart và không có process mới do thay đổi này. |
| 08/10/2026 | ✅ Bounded runtime telemetry | Latency telemetry 24/7 không còn giữ list tăng vô hạn: pipeline dùng rolling deque tối đa 10.000 translation hoàn tất, công bố sample count/capacity/dropped count và giữ counter tổng độc lập. Mọi đường thành công, gồm cache hit dùng audio dựng sẵn, đều đi qua một recorder; cache/audio validation exception không còn tăng `cache_hits` hoặc `total_translations`, nên hit rate không bị đếm false success. P95 sửa sang nearest-rank `ceil(0,95×n)-1` thay vì lệch một mẫu; min/max/average cùng đọc snapshot bounded dưới runtime lock. Regression đẩy quá capacity, đối chiếu retained range/drop count/p95 và kiểm cả cache success/failure. Full local suite: 401 pass, 1 skip. |
| 08/10/2026 | ✅ Pipeline concurrency isolation | Một `MediVoicePipeline` nay serialize toàn bộ operation hữu hạn chia sẻ model/counter/audio sink: component load, speech/text translation, playback và performance snapshot. Reentrant lock cho phép operation pipeline lồng nhau cùng thread mà không deadlock; exception luôn nhả lock. Microphone session có lock riêng non-blocking: session thứ hai fail-fast thay vì reset/chia sẻ buffer, nhưng `get_status()` vẫn đọc được khi phiên live kéo dài. Regression instrumented xác minh boundary giữ lock, lời gọi lồng đạt depth 2 rồi nhả sạch, stats snapshot nhất quán, duplicate interactive bị chặn và validation failure nhả session lock. Full local suite: 399 pass, 1 skip. |
| 08/10/2026 | ✅ MT concurrent-direction isolation | `MTEngine` nay khóa một critical section duy nhất từ lúc gán `tokenizer.src_lang` qua tokenize, device transfer, generation, EOS validation đến decode, nên request VI→EN và EN→VI đồng thời không thể đổi hướng tokenizer dùng chung hoặc chạy chồng model state. Thời gian chờ lock được tính vào latency thực. Load/reload hạ readiness và publish model/tokenizer/path dưới cùng lock để không tráo runtime object giữa inference. Regression dùng lock instrumented xác minh mọi shared-state/model operation nằm trong đúng một section và exception model luôn nhả lock, không deadlock request sau. Full local suite: 396 pass, 1 skip. |
| 08/10/2026 | ✅ Component runtime readiness | Các API component dùng trực tiếp nay cũng fail-closed theo readiness đầy đủ: ASR cần đủ model/processor/model-path cho mọi ngôn ngữ cấu hình trước cả detect/transcribe; MT cần model/tokenizer/model-path; TTS cần đồng thời hai voice; Flash Cache chưa load hoặc cache rỗng không còn bị hiểu nhầm là cache miss. Regression cố ý đặt `_is_loaded=true` nhưng để state nửa vời và xác minh cả detect, transcribe, translate, synthesize, lookup đều dừng tại boundary thay vì chạm tokenizer/model/voice/cache. Các unit fixture cũ được nâng lên contract đầy đủ, không hạ production guard. Full local suite: 394 pass, 1 skip. |
| 08/10/2026 | ✅ Runtime readiness enforcement | `translate_speech`, `translate_text` và `run_interactive` không còn tin cờ pipeline loaded cũ; mỗi request đọc readiness động của AudioFrontend/ASR/MT/TTS/cache và fail-closed với danh sách component thiếu trước validation/model/cache/device side effect. Logic xác định component bắt buộc dùng chung với startup gate, gồm ngoại lệ cache-disabled duy nhất. Regression đặt cố ý `_is_loaded=true` nhưng làm TTS unready và xác minh cả ba entrypoint cùng chặn trước runtime work. Các test pipeline cũ nay dùng fixture dựng đủ dummy model/processor/tokenizer/two voices/default cache thay vì giả readiness bằng một boolean. Full local suite: 393 pass, 1 skip; compile/dependency/diff checks pass. |
| 08/10/2026 | ✅ Atomic ASR/MT model loading | ASR nay resolve device và load processor/model VI+EN vào staging; chỉ sau khi cả hai model `.to(device)` và `.eval()` thành công mới publish đồng thời device, dictionaries và model-path evidence. Lỗi EN không còn để lại VI đã publish. MT tương tự: tokenizer/model/path/device chỉ publish sau candidate hoặc fallback load, `.eval()` và parameter inspection hoàn tất; failure giữ nguyên state chưa ready. Fallback policy vẫn không đổi và mọi configured-checkpoint failure khi fallback bị tắt giữ exception cause. Regression mô phỏng failure giữa chừng cùng full local suite: 392 pass, 1 skip; compile/dependency/diff checks pass. Không tuyên bố model artifact cold-start thật vì test này dùng deterministic fake loaders. |
| 08/10/2026 | ✅ Pipeline startup/readiness hardening | Bỏ việc tin một cờ `_is_loaded` tự gán: AudioFrontend, ASR, MT, TTS và Flash Cache nay có readiness contract kiểm tra đủ backend/model/tokenizer/model-path/hai voice/cache runtime. Orchestrator công bố `get_status()` theo từng component, chỉ set pipeline loaded sau readiness gate và nêu đúng stage fail; cache disabled được loại khỏi tập bắt buộc. `load()` đã verified trở thành idempotent nên không reload/nhân đôi model; một retry chưa ready luôn hạ cờ pipeline trước khi bắt đầu. Piper VI/EN load vào staging và chỉ publish đồng thời, lỗi voice thứ hai không để lại voice thứ nhất dưới trạng thái nửa vời. Regression local: 390 pass, 1 skip; compile/dependency/diff checks pass. Đây là contract/mocked failure evidence, không thay thế cold-start thật với toàn bộ model artifact. |
| 08/10/2026 | ✅ CLI lifecycle/exit hardening | CLI nay validate mode-specific args trước model setup: file mode bắt buộc path tồn tại và không vượt 64 MiB, text mode cấm blank, benchmark chỉ nhận 1–10 mẫu. File audio được inspect/decode đầy đủ trước `pipeline.load()`, nên metadata/codec/duration/allocation lỗi không còn tốn thời gian hoặc RAM để load model trước. Benchmark số lẻ phân bổ VI/EN và chạy đúng đủ N thay vì âm thầm thiếu một mẫu. Entrypoint trả ổn định code 2 cho syntax/argument, 1 cho lỗi vận hành, 130 cho interrupt ngoài interactive; lỗi dự kiến có thông báo nhưng không traceback, lỗi bất ngờ chỉ lộ tên exception và che chi tiết có thể nhạy cảm. Subprocess smoke thật xác nhận missing `--input` exit 2 và config thiếu exit 1 không traceback. Regression local: 385 pass, 1 skip; compile/dependency/diff checks pass. |
| 08/10/2026 | ✅ Audio-output safety sink hardening | `MediVoicePipeline.play()` nay tái kiểm tra `safety_passed` và `requires_confirmation` ngay tại sink, nên một result bị sửa/ghép sai ở caller vẫn không thể phát candidate fail clinical gate hoặc nội dung chưa xác nhận. `TTSEngine.play_audio()` vẫn revalidate waveform trước dispatch và nay bảo đảm thử `sounddevice.stop()` nếu `play/wait` bị lỗi, `KeyboardInterrupt` hoặc hủy giữa chừng, đồng thời re-raise nguyên exception gốc; lỗi cleanup phụ chỉ log loại lỗi, không log audio/text. Regression local: 375 pass, 1 skip; compile/dependency/diff checks pass. Loa/driver vật lý chưa được benchmark nên không phải hardware evidence. |
| 08/10/2026 | ✅ Live microphone orchestration hardening | Sửa live path từng denoise cùng audio hai lần: segment đã qua noise suppression để VAD nay được đánh dấu explicit, vẫn revalidate audio/window nhưng bỏ lần suppression thứ hai trước ASR; file/batch audio vẫn phải đi qua frontend như cũ. Interactive path truyền đúng sample rate cấu hình thay vì ngầm 16 kHz, từ chối marker sai kiểu hoặc sample rate không khớp. CLI nay nhận từng kết quả và hiện transcript, bản dịch hoặc nhãn hard-block cùng latency/confidence/cache trước khi playback; candidate fail clinical không bị lộ và `KeyboardInterrupt` chỉ được xử lý ở biên CLI sau khi context microphone đóng. Regression local: 373 pass, 1 skip; compile/dependency/diff checks pass. Microphone/PortAudio vật lý vẫn chưa được benchmark nên không phải hardware evidence. |
| 08/10/2026 | ✅ Microphone streaming hardening | Đường microphone không còn dùng `deque(maxlen=100)` có thể âm thầm vứt audio cũ: callback nay ghi vào queue thread-safe giới hạn khoảng 2 giây, bảo toàn đủ channel và fail-closed khi PortAudio báo lỗi, frame count/shape/sample sai hoặc consumer chậm gây overflow. Consumer dùng blocking poll có timeout thay vì busy-wait. Segmenter giới hạn tích lũy liên tục bằng đúng `asr.max_input_duration_seconds` (canonical 30 giây), reset state và chặn nếu speech/pause vượt cửa sổ thay vì tăng RAM vô hạn hoặc chuyển audio bị Whisper cắt. Cấu hình không hữu hạn, không dương hoặc ngắn hơn một chunk bị từ chối lúc khởi tạo. Regression local: 369 pass, 1 skip; physical microphone/PortAudio thật chưa được benchmark nên không được coi là hardware evidence. |
| 08/10/2026 | ✅ Offline audio-frontend hardening | Loại bỏ hoàn toàn `torch.hub` Silero download và energy fallback ngầm khỏi runtime. VAD nay dùng `webrtcvad-wheels==2.0.14.post1` đã pin (MIT, wheel CPython đa nền tảng), chạy local-only trên PCM16 mono 10 ms, cấu hình aggressiveness 0–3 và trả voiced-frame ratio; dependency thiếu, sample rate sai hoặc inference lỗi đều fail-closed. Noise suppression có cờ boolean explicit: canonical config tắt và readiness báo pass-through rõ ràng; nếu bật mà thiếu backend hoặc backend lỗi thì chặn, không còn âm thầm trả raw audio. Microphone giữ đủ channel tới validator/downmix thay vì lấy kênh 0. Smoke thật local: frontend `ready=true`, silence ratio 0, tone ratio 0,8, `pip check` sạch. Regression local: 360 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` và `git diff --check` pass. Denoiser đã validation vẫn là blocker riêng trước clinical live demo. |
| 08/10/2026 | ✅ CLI audio-ingress hardening | File mode nay đọc metadata trước khi decode và fail-closed nếu path thiếu, file/decoded allocation vượt 64 MiB, không có frame, sample rate ngoài 8–384 kHz, quá 8 kênh, duration vượt ASR window 30 giây hoặc metadata không khớp decoded shape/rate. Nhờ đó file quá dài/khổng lồ dừng trước cấp phát; stereo không còn bị âm thầm bỏ kênh phải mà được giữ nguyên để pipeline validation downmix đúng sau khi kiểm tra peak từng kênh. `soundfile` decode thẳng float32 có shape contract. ASR confidence proxy NaN/Inf/ngoài `[0,1]` bị chặn và CLI ghi rõ đây là token-probability proxy chưa hiệu chuẩn, không phải bảo đảm lâm sàng. Regression local: 351 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` và `git diff --check` pass. |
| 08/10/2026 | ✅ Clinical output/privacy hardening | User-facing orchestrator nay không phát hành candidate MT khi hard safety gate fail: `translated_text` bị xóa, TTS không được gọi, còn engine MT thấp tầng vẫn giữ candidate + evidence để benchmark/chẩn đoán. CLI chỉ hiện nhãn `BLOCKED BY CLINICAL SAFETY GATE` cùng issue category đã khử chi tiết; action cache hợp lệ vẫn hiện cảnh báo cần xác nhận trước playback. ASR rỗng không còn mặc định `safety_passed=true` mà fail-closed với `empty_asr_transcript` trước cache/MT/TTS. Runtime log ASR/MT/cache/orchestrator và lỗi pre-synthesis chỉ ghi language, độ dài, latency, loại lỗi/safety category; không còn ghi raw transcript, source hay translation, đúng privacy contract. Regression local: 344 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` và `git diff --check` pass. |
| 08/10/2026 | ✅ Pipeline configuration hardening | Pipeline config explicit nay fail-closed khi file thiếu/quá 1 MB/YAML lỗi hoặc trùng key, root/section sai kiểu, section hay field lạ/thiếu, boolean dạng chuỗi, topology khác đúng hai chiều VI↔EN, hoặc bật streaming/speculative decoding/fuzzy cache chưa được hỗ trợ. `bit_depth` và `channels` được truyền thật vào audio frontend; cache `enabled: false` không còn load hoặc lookup. MT runtime khóa temperature 0 vì decoding không sampling. Medical lexicon explicit nay bắt buộc tồn tại, giới hạn 1 MB, từ chối JSON trùng key/sai schema/giá trị rỗng và chỉ publish nguyên tử sau khi toàn bộ hai chiều hợp lệ. Regression local: 340 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` và `git diff --check` pass. |
| 08/10/2026 | ✅ ASR window/generation hardening | Runtime và benchmark ASR nay từ chối audio vượt cửa sổ Whisper 30 giây trước denoise/feature extraction thay vì để feature extractor cắt âm thầm; input benchmark cũng đi qua validation finite/normalized/channel-layout trước downmix. Mọi transcript phải sinh EOS thật sau decoder-start token và không được có token nội dung sau EOS trước decode/scoring. Prediction checkpoint ASR khóa context 30 giây, output cap 225 và `require_eos`, nên artifact cũ thiếu contract không được resume; contract MT không bị ảnh hưởng. Auto language detection không còn nuốt lỗi rồi mặc định `vi`: engine chưa load, thiếu token VI/EN, logits NaN/Inf/đồng hạng hoặc inference lỗi đều fail-closed. ASR/MT dùng chung một generation guard để giữ cùng semantics. Regression local: 323 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` pass. |
| 08/10/2026 | ✅ Emergency cache hardening | Flash Cache nay load default + custom theo staging nguyên tử; file custom đã cấu hình mà thiếu, quá 1 MB, quá 1.000 mục, JSON/schema/type/direction sai, field lạ, source trùng sau normalize hoặc vi phạm number/unit/negation safety đều fail-closed, không còn nuốt lỗi hay publish một phần. Custom phrase luôn bắt buộc confirmation; fuzzy matching bị vô hiệu hóa. Normalization chỉ bỏ terminal punctuation nên `5.0 mg` không còn va chạm nguy hiểm với `50 mg`; orchestrator đối chiếu cache direction trước khi bypass MT. Đồng thời sửa guard để phân biệt `không` cuối polar question tiếng Việt với phủ định thật, giúp toàn bộ emergency cache mặc định qua structural safety mà vẫn bắt `Không dùng...`. Xóa placeholder custom-cache không tồn tại khỏi config. Regression local: 313 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` pass. |
| 08/10/2026 | ✅ TTS output-boundary hardening | Piper/TTS nay fail-closed với text không phải chuỗi, rỗng, quá 4.096 ký tự hoặc chứa control character/NUL; cấu hình khóa output 22.050 Hz và tối đa 120 giây. Mỗi chunk phải là mono float32 hữu hạn, không rỗng, nằm trong `[-1,1]`, có sample rate nguyên 8–384 kHz và nhất quán; output sau resample được kiểm tra lại. Chunk generator được tiêu thụ có duration cap tức thời thay vì materialize vô hạn. Cached audio được revalidate trước khi rời orchestrator, và `play_audio` kiểm tra lại ngay trước device sink. Regression local: 305 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` pass. |
| 08/10/2026 | ✅ MT generation-completion hardening | Runtime và benchmark MT nay bắt buộc mỗi output sinh EOS kết thúc thật sau decoder-start token; EOS đầu chuỗi của họ NLLB không được tính nhầm là hoàn tất. Nếu model chạm `max_new_tokens` hoặc trả chuỗi token malformed/chứa token nội dung sau EOS, pipeline fail-closed trước decode, scoring hoặc TTS. Prediction checkpoint MT khóa thêm inference contract gồm context 256, output cap 256, không truncate source và bắt buộc EOS; checkpoint MT cũ thiếu contract bị từ chối, trong khi schema/checkpoint ASR không bị vô hiệu hóa ngoài phạm vi. Regression local: 293 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` pass. |
| 08/10/2026 | ✅ MT truncation/output hardening | Runtime và benchmark MT không còn dùng `truncation=True`: tokenizer đọc toàn bộ source rồi fail-closed nếu vượt context 256 token, thay vì âm thầm bỏ phần cuối có thể chứa liều hoặc phủ định. Runtime cũng chặn source không phải chuỗi, rỗng hoặc vượt 4.096 ký tự trước cache/model dispatch. Safety guard gắn `empty_translation` cho output rỗng; lỗi invariant này được tính vào mọi clinical slice nên không thể pass riêng `drug_name`, `dose`, `number`, `unit`, `negation`, `terminology` hay `code_switch`. Regression local: 290 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` và `git diff --check` pass. |
| 08/10/2026 | ✅ Audio/ASR boundary hardening | Thêm validation chung trước denoiser, VAD và ASR: từ chối audio rỗng, NaN/Inf, biên độ ngoài `[-1,1]`, ndim không hợp lệ, channel-first/khối đa kênh mơ hồ, hoặc sample rate không nguyên/nằm ngoài 8–384 kHz. Stereo/channel-last hợp lệ vẫn downmix và resample 16 kHz như cũ; kiểm tra peak diễn ra trước downmix để hai kênh out-of-range không triệt tiêu nhau. `AudioConfig` nay fail-fast với sample rate, bit depth, channel count, chunk/silence duration và VAD threshold sai. Regression local: 286 pass, 1 skip; compile pass. |
| 08/10/2026 | ✅ Clinical runtime hardening | Safety guard nay khóa quan hệ giữa từng thuật ngữ và liều/phủ định, nên không còn bỏ lọt trường hợp hai thuốc giữ nguyên tổng multiset nhưng bị tráo liều, hoặc số lượng phủ định không đổi nhưng phủ định bị chuyển sang thuốc khác. Quan hệ liều dùng nearest unambiguous term; quan hệ phủ định dùng strong clause scope và fail-closed khi quantity binding mơ hồ. `score_mt` đưa lỗi mới vào đúng hard gate `dose`/`negation`; lexicon runtime bổ sung các thuốc đã khóa trong clinical suite. `MTEngine.translate`, text-only API và cache hit cùng trả `safety_passed`, `safety_issues`, `requires_confirmation`, không còn bypass safety metadata. Toàn bộ 16 reference MT pass guard theo cả hai chiều. Regression local: 283 pass, 1 skip; compile pass. |
| 08/10/2026 | ✅ Bake-off resume hardening | Sửa nguyên nhân recovery dừng ở completed MT training stage: stage train cũ không mang `--bakeoff-runner-sha256`, nên logic tương thích 829940e không xét được trường hợp chỉ khác cách viết interpreter (`../.venv-onevoice/bin/python` so với `./.venv-onevoice/bin/python`). Resume nay chỉ chấp nhận alias khi hai đường dẫn `resolve(strict=True)` tới cùng regular file và mọi token còn lại giống byte-for-byte; mọi thay đổi argument khác vẫn bị từ chối. Completed stage chỉ được skip khi có `output_evidence` đã ghi và bằng chứng hiện tại khớp. Regression local: 275 pass, 1 skip; compile toàn bộ `src/scripts/demo/tests` pass. Không thay đổi hoặc restart process remote. |
| 27/09/2026 | ✅ Accuracy program | Khóa kế hoạch `configs/accuracy_program.yaml`: chọn checkpoint theo WER/BLEU/chrF nhưng bắt buộc mọi slice thuốc, liều, số, đơn vị, phủ định, thuật ngữ và code-switch cùng pass. Thêm train-only clinical oversampling hệ số 2; validation/test không nhân bản và suite lâm sàng 16 cặp/32 chiều bị kiểm tra exact-pair không trùng train. |
| 27/09/2026 | ✅ Local fine-tune smoke | Sửa training CLI hỗ trợ LoRA và chế độ CPU smoke fail-closed. Chạy thật PhoWhisper một bước: 1.769.472 tham số trainable, train/eval loss 2,5475/2,3259. Chạy thật NLLB joint một bước: 2.359.296 tham số trainable, train/eval loss 1,3301/2,2388. Cả hai artifact ghi `promotion_allowed=false`; đây chỉ chứng minh đường train hoạt động, không phải bằng chứng accuracy tăng. |
| 27/09/2026 | ⛔ Clinical release gate | NLLB base benchmark trên suite khóa 32 chiều đạt BLEU 46,29/chrF2 63,46 nhưng safety gate fail: thuốc/liều/số/đơn vị 0 lỗi phát hiện, phủ định 5,56%, thuật ngữ 33,33%, code-switch 75%. Aggregate baseline cũng chưa đạt ngưỡng (ASR VI 20,87% >19%, ASR EN 25,54% >23%, MT BLEU 23,76 <25). `release_gate.json` đặt `promotion_allowed=false`; ví dụ nguy hiểm thật: `HIV negative` bị dịch thành `HIV dương tính`. |
| 22/09/2026 | ✅ Done | Xác nhận dự án chỉ có một người; bỏ toàn bộ phân công team cũ. |
| 22/09/2026 | ✅ Done | Audit repository và notebook merge cũ; giữ ý tưởng đa nguồn nhưng thay cơ chế merge gây leakage. |
| 22/09/2026 | ✅ Done | Sửa lựa chọn ASR Việt từ English-only `distil-small.en` sang multilingual `vinai/PhoWhisper-small`. |
| 22/09/2026 | ✅ Done | Khóa Eka English ASR làm evaluation-only; không dùng split `test` để fine-tune. |
| 22/09/2026 | ✅ Done | Thay causal chat LLM cho MT bằng encoder-decoder NLLB baseline; tắt distillation/chain-of-thought trước khi có baseline. |
| 22/09/2026 | ✅ Done | Thêm registry dữ liệu, EDA, leakage checks, merge gate, audio materializer và training preflight. |
| 22/09/2026 | ✅ Done | Regression suite tại thời điểm chốt ban đầu có 32 tests pass; toàn bộ Python trong `src/scripts/tests/demo` compile pass. |
| 22/09/2026 | ✅ Done | Full metadata EDA: 9.207 VietMed, 15.818 ViMedCSS, 358.796 cặp MedEV và 3.619 Eka. |
| 22/09/2026 | ✅ Done | Phát hiện và định lượng leakage: ViMedCSS có 102 transcript và 1.413 video chéo role; MedEV ghép đúng có 41 cặp train↔validation và 48 train↔test; merge gate đã khóa pair + speaker + recording-group. |
| 22/09/2026 | ✅ Done | Tải Piper VI/EN, chạy CPU smoke thật cho ASR/MT/TTS; sửa resample ASR 22,05 kHz→16 kHz và bỏ fallback TTS im lặng. |
| 22/09/2026 | ✅ Human gate | Chủ dự án xác nhận audio tốt và transcript khớp 100%; signal QC đã thay hai clip rác. Không cần label bổ sung ở vòng này. |
| 22/09/2026 | ✅ Fixed | Chủ dự án phát hiện listening pack đầu chứa quá nhiều clip 0,48–1 giây/khó nghe. Root cause: risk flags chiếm hết quota. Đã cân bằng quota và dùng gate **riêng cho pack nghe** `duration>=2s`, `RMS>=0.003`; tạo playback mono 16 kHz PCM16 chuẩn hóa âm lượng và trang `review.html`. Dataset chính không loại máy móc mọi clip dưới 2 giây, chỉ loại clip thực sự rác/không nghe được theo signal gate. Hai pack cũ được chuyển vào `data/review/.trash/` để có thể phục hồi. |
| 22/09/2026 | ✅ Owner decision | Chủ dự án cho phép dùng MedEV trong phạm vi research/non-commercial, không tái phân phối; 89 exact pair raw chéo split vẫn phải bị loại khỏi role thấp hơn khi merge. |
| 22/09/2026 | ✅ Human QA | Chủ dự án xác nhận transcript trong listening pack khớp 100%; 12/12 mẫu mỗi nguồn được ghi `audio_ok=yes`, `transcript_exact=yes`, `keep`. |
| 22/09/2026 | ✅ PII audit | Scan lại 399.860 records sau sửa pairing và bổ sung VIVOS: 270 candidates; đa số là DOI/SNP/bảng số hoặc câu chung nhắc “địa chỉ”. Loại 3 records có bằng chứng rõ: tên bệnh nhân hoặc email/điện thoại; 267 false positives được review và checksum-lock tại `configs/data_review_decisions.yaml`. |
| 22/09/2026 | ✅ Merge | Giữ manifest strict-official để đối chiếu; manifest primary repartition ViMedCSS theo video, khóa hard groups, tăng ASR train từ 9,315h lên 28,443h mà vẫn 0 group/speaker leakage. |
| 22/09/2026 | ✅ Materialized | Resume-aware materializer hoàn tất 32.055 raw assets theo manifest; 12 upstream empty assets đã loại có version, không còn download error/missing ID. |
| 22/09/2026 | 🚨 Corrected | Content inspection bắt lỗi MedEV: raw split không xen kẽ; nửa đầu là EN, nửa sau là VI aligned theo index. Hủy manifest MT sai, sửa pairing theo `i ↔ i+n/2` và chạy lại full audit/PII/merge trước GPU. |
| 22/09/2026 | ✅ MT audit | Audit sâu 358.796 cặp MedEV đúng: median tỷ lệ ký tự VI/EN ≈0,98; giữ `number_set_mismatch` làm review slice, loại 854 cặp có sai ngôn ngữ/identity, length-ratio cực đoan hoặc quá dài. Manifest MT cuối 339.028/8.929/8.938, exact-pair overlap chéo role = 0. |
| 22/09/2026 | ✅ Training design | Sửa MT từ hai checkpoint lệch runtime thành một NLLB joint EN↔VI (678.056 effective train examples, preflight pass); ASR chuyển sang augmentation thật theo batch: speed/gain/noise, validation không augment. |
| 22/09/2026 | ✅ Diversity EDA | VIVOS full audit pass: 11.660 train + 760 calibration, 46/19 speakers, 0 speaker/group leakage. Giới hạn tối đa 80 câu/train speaker; owner đã xác nhận pack nghe riêng 12/12 khớp. ViMD 63 tỉnh được ghi làm accent benchmark tương lai vì full source ≈59,8 GB và CC BY-NC-ND có rủi ro derivative. |
| 22/09/2026 | ✅ Baseline MT | NLLB base CPU benchmark thật trên 24 câu hai chiều: BLEU 23,763; chrF2 44,963. EN→VI BLEU 24,631; VI→EN 22,429. Sửa guard false-positive cho dấu thập phân có khoảng trắng, digit↔number-word, implicit negation và `e.g.`; safety failures sau sửa = 0/24. |
| 22/09/2026 | ✅ Audio exclusion | 12 ViMedCSS records trả asset rỗng sau 4 lần retry; đã version hóa toàn bộ ID trong `configs/data_exclusions.yaml`, remerge và validation pass. |
| 22/09/2026 | ✅ Signal QC | Decode 32.055 assets; sau signal gate giữ 17.849 train, 3.733 validation, 10.393 test. Loại 77 near-silence + 3 decoded CPS bất khả thi; 1 mẫu đồng thời lệch duration lớn. |
| 22/09/2026 | 🚨 Audio-hash leakage fixed | SHA-256 EDA phát hiện 46 nhóm FLAC trùng bit của ViMedCSS dù ID/video khác, trong đó 17 nhóm chéo model role. Giữ role cao nhất (`test > validation > train`), loại 43 train + 3 duplicate-test; final 17.806/3.733/10.390 với 31.929 hash duy nhất và cross-role audio overlap = 0. |
| 22/09/2026 | ✅ CPU baselines | PhoWhisper VI: WER 20,87%, CER 15,39% (24 mẫu); Distil-Whisper EN: WER 25,54%, CER 7,85% (24 mẫu); NLLB joint: BLEU 23,76, chrF2 44,96 (24 mẫu). Artifact có prediction từng câu và bootstrap WER CI. |
| 22/09/2026 | ✅ GPU gate | `preflight_gpu_ready.json` ready=true; regression suite pass; artifact revisions/checksums khóa; `GPU_HANDOFF.md` có dry-run 100 step, full commands, resume, budget và shutdown policy. |
| 22/09/2026 | ✅ Done | Materialize/signal-QC audio, benchmark locked set và đóng gói GPU handoff đã hoàn tất; GPU preflight `ready=true`. |
| 22/09/2026 | ✅ Repo audit | Đồng bộ README/trạng thái; proposal/research cũ không được dùng làm cấu hình hay nguồn sự thật vì còn claim chưa đo và mô tả team nhiều người. Mọi báo cáo vận hành tập trung tại file này. |
| 22/09/2026 | ✅ Legacy safety | Downloader cũ được thay bằng snapshot tool đọc registry + revision lock, không cho bulk-download/nguồn disabled và khóa Eka sau cờ evaluation-only. QNN scaffold dùng NLLB seq2seq exporter, chỉ cho INT8 candidate và fail-closed với INT4 thay vì giả UINT8 là INT4. |
| 22/09/2026 | ✅ Owner correction | Theo yêu cầu chủ dự án, hoàn tác toàn bộ thay đổi DOCX/generator: hai DOCX được khôi phục byte-for-byte đúng Git blob gốc (`4776dd3…`, `b532409…`). Không ghi báo cáo vào DOCX; mọi trạng thái/nhận xét chỉ cập nhật trong `project_analysis.md`. |
| 22/09/2026 | ✅ Transfer integrity | Sửa GPU gate từ “chỉ in hash” sang so sánh thật 6 manifest với artifact lock. Inventory cuối khóa 31.929 FLAC duy nhất; cloud preflight băm lại từng file và fail nếu thiếu/sai size/sai hash. Sau DC repair, inventory là 3.541.434.742 bytes và SHA-256 `279c1014…`. |
| 22/09/2026 | ✅ Final CPU verification | Compile toàn bộ `src/scripts/tests/demo` pass; `pytest` 32/32 pass; GPU preflight xác minh đủ 31.929 file cùng 6 manifest đúng artifact lock. |
| 22/09/2026 | ✅ VIVOS human gate | Chủ dự án nghe pack bổ sung và xác nhận 12/12 audio–transcript khớp 100%. `review.csv` đã ghi `audio_ok=yes`, `transcript_exact=yes`, `language=vi`, `pii=no`, `keep`; audit VIVOS chuyển sang `reviewed`. |
| 22/09/2026 | ✅ Pre-GPU hardening | Pin commit revision vào ASR/MT loader và bắt GPU gate so khớp artifact lock. Flash cache chuyển exact-only mặc định; câu hành động lâm sàng luôn yêu cầu xác nhận trước TTS. Regression 32/32 pass. |
| 22/09/2026 | ✅ Signal repair | Final-audio EDA tìm thấy 2 ViMedCSS train clip có DC offset >0,10 nhưng lời nói vẫn hợp lệ. Không xóa dữ liệu: trừ DC bias, giữ waveform lời nói, cập nhật `audio_qc.repairs`, băm lại toàn bộ. Inventory vẫn 31.929 hash duy nhất, 0 overlap; train manifest SHA mới `0be3b56d…`. |
| 22/09/2026 | ✅ Accent-source expansion audit | Tìm thêm nguồn có nhãn vùng. FPTU Vovinam full metadata: 17.560 rows/367 train speakers nhưng 13.596/14.048 train là repeated-text rows, 94,9% Central và published split rò 323 train↔validation + 312 train↔test speakers; không merge. ViVoice34 chỉ là 381-row anonymous preview với clip 20–113s; GovVox cần manual access/license `other`. Cả ba được registry hóa nhưng khóa mặc định. |
| 22/09/2026 | ✅ Augmentation hardening | Thay Gaussian-only nhẹ bằng train-only stochastic profile có speed/gain, white–pink–mains-hum noise SNR 12–35 dB, synthetic short reverb, telephone band-limit và μ-law codec simulation. Validation/test không augment; seed/profile được preflight ghi rõ và có deterministic unit test. |
| 22/09/2026 | ✅ MT token-length EDA | Đo bằng đúng NLLB tokenizer/revision trên toàn bộ 356.895 cặp locked. Tăng context 192→256; tỷ lệ vượt giới hạn cao nhất còn 0,1456% (validation VI), mọi split/hướng dưới gate 0,20%. Artifact: `data/reports/eda/medev_token_lengths.json`. |
| 22/09/2026 | ✅ Quantization safety | ONNX Runtime dynamic-QInt8 được ghi rõ chỉ là desktop parity prototype, không phải calibrated QNN INT8. Yêu cầu SNPE DLC nay fail-closed nếu thiếu SDK/converter lỗi/không sinh output; không còn dùng QNN converter rồi gắn nhãn sai `.dlc`. QCS6490 QNN calibration/context binary vẫn để sau checkpoint GPU thật. |
| 22/09/2026 | ✅ Runtime checkpoint integrity | Base-model fallback trước GPU nay là cờ cấu hình hiện rõ, không còn mặc định ngầm trong engine. Sau khi sync checkpoint GPU phải đổi `runtime.allow_base_model_fallback=false`; checkpoint thiếu/hỏng sẽ chặn startup thay vì khiến demo chạy nhầm base model. |
| 22/09/2026 | ✅ Final stop gate reached | Completion audit cuối: compile pass, `pytest` 32/32, `pip check` sạch, `git diff --check` không lỗi, ASR/MT BF16 100-step preflight pass và GPU gate `ready=true` sau khi băm lại 31.929 FLAC. Hai DOCX vẫn đúng Git blob gốc `4776dd3…` / `b532409…`. Không còn label hoặc bước CPU bắt buộc; bước tiếp theo là thuê GPU và chạy dry-run có tính phí. |
| 22/09/2026 | ✅ Tài liệu dễ đọc hơn | Thêm nghĩa, đầu vào → đầu ra, mục tiêu và lý do cho ASR/MT/TTS cùng từng phase dữ liệu–đánh giá–triển khai. **Lý do:** chủ dự án cần đọc nhanh mà không phải tự giải mã thuật ngữ kỹ thuật. Không thay đổi dữ liệu, code hay trạng thái GPU-ready. |
| 22/09/2026 | ✅ Full base cascade smoke | Đính chính: smoke cũ chỉ đo từng model, chưa nối output của ASR sang MT rồi TTS. Đã chạy thêm hai audio thật đã khóa theo đủ chuỗi `audio → base ASR → base MT → safety → Piper TTS`, một chiều VI→EN và một chiều EN→VI; cả hai tạo WAV thành công. **Lý do:** chứng minh các module ghép được với nhau trước GPU, đồng thời thấy rõ lỗi ASR làm bản dịch xấu đi. |
| 22/09/2026 | ⛔ Stop gate | Chỉ dừng để báo chủ dự án khi mọi preflight đã qua và bước kế tiếp thực sự cần GPU fine-tune. |
| 25/09/2026 | ✅ Full local rerun + hardening | Chạy lại toàn bộ chuỗi local an toàn: `pytest` 40/40, compile pass, `pip check` sạch, manifest/training preflight pass, ASR VI/EN + MT baseline tái lập đúng số cũ, component smoke/full cascade/CLI text pass và GPU gate vẫn `ready=true` sau khi băm lại đủ 31.929 FLAC. Sửa guard phát hiện tráo cặp số–đơn vị, cache direction validation, VAD silence rounding, hai lỗi stdout UTF-8 trên Windows và CPU smoke trỏ nhầm checkpoint chưa tồn tại. Alignment audit nay sắp thứ tự flag deterministic; chạy hai process cho cùng SHA-256 rồi cập nhật checksum quản trị, không thay đổi nội dung quyết định hoặc manifest/audio lock. Fine-tune thật vẫn bị chặn đúng vì máy chỉ có PyTorch CPU, không có CUDA. |
| 25/09/2026 | ✅ Local decoding experiments | Chạy ablation trên locked validation bằng đúng base model local. ASR greedy WER 19,40%; beam-4 chỉ còn 18,99% nhưng chậm khoảng 3 lần và làm xấu hai slice Doctor/Tel nên không chọn. Prompt y khoa tĩnh đạt WER 18,17% nhưng gây lỗi nguy hiểm `viêm tuyến Bartholin → viên tuyến insulin`, vì vậy bị loại. MT greedy đạt BLEU/chrF2 25,93/46,02; beam-4 giảm còn 24,13/45,26 và chậm khoảng 4,8 lần. Locked test MT greedy đạt BLEU/chrF2 23,26/45,24 so với beam-4 cũ 23,76/44,96; chọn greedy vì validation tốt hơn rõ, tốc độ cao hơn nhiều và safety cùng 0%. Runtime nay đọc `mt.num_beams: 1` từ config có regression test; benchmark runner lưu beam/prompt/thời gian để tái lập. Guard cũng chuẩn hóa mã y khoa tương đương `T3`/`T 3`, `COVID-19`/`COVID19` nhưng vẫn chặn thay đổi thật `T3 → T4`. |
| 25/09/2026 | ✅ Budget GPU plan | Bỏ A100 80 GB khỏi lựa chọn mặc định vì chi phí cao. Handoff chuyển sang A40/RTX A6000 48 GB FP16, train/eval batch 4 và gradient accumulation 8; fallback 24 GB dùng batch 2/2 + accumulation 16, vẫn giữ effective batch 32 và context/data đầy đủ. CLI ASR/MT nay cho chỉnh riêng validation batch và gradient accumulation. Dry-run đầu đặt budget alert USD 10; A100 chỉ còn fast path tùy chọn. |
| 25/09/2026 | ✅ AI Hub planning | Qualcomm AI Hub CLI 0.55.0 đã xác thực account và nhìn thấy `Dragonwing RB3 Gen 2 Vision Kit` / QCS6490 / Qc_Linux 1.6. Framework hiện có QAIRT 2.45, 2.49 và 2.50 (`default`, `latest`). Tạo `AIHUB_DEPLOYMENT_PLAN.md` với gate theo thứ tự catalog smoke → PhoWhisper → Piper → NLLB → quyết định kiến trúc → GPU fine-tune → INT8 regression → board acceptance; mục tiêu là phát hiện model không deploy được trước khi trả tiền train. |
| 26/09/2026 | ✅ AI Hub reference asset | Cài `qai-hub-models-cli` 0.63.0, khóa môi trường/QAIRT 2.50, tải Whisper Small Quantized w8a16 QNN context binary chính thức cho QCS6490 và ghi SHA-256. Catalog báo encoder 233,30 ms + decoder 23,46 ms/token trên NPU; 20 token ước lượng 702,50 ms nên ngân sách ASR 650 ms hiện chưa đạt. |
| 26/09/2026 | ✅ AI Hub hosted profile | Cả encoder và decoder reference đều profile thành công trên Dragonwing RB3 Gen 2/Qc_Linux 1.6 (QCS6490). Decoder p50 `34,089 ms/token`, peak `74,65 MB`, 2.853 ops NPU. Encoder `jgoldyjxg` (không truyền profile option) p50 `239,419 ms`, peak `36,23 MB`, 1.894 ops NPU. Tổng thời gian 20 token p50 ≈921 ms; vượt ngân sách ASR 650 ms nhưng chứng minh Whisper-small reference deployable toàn bộ lên NPU. A/B gợi ý lỗi cũ liên quan options/server, chưa chứng minh riêng `--compute_unit` là nguyên nhân duy nhất. |
| 26/09/2026 | ✅ Edge fallback decision | Catalog không có Piper QCS6490 và không có MT Việt–Anh/NLLB; do đó Piper VI/EN giữ CPU baseline (RTF 0,0591/0,0772), còn NLLB là BYOM fail-closed. Full `qai-hub-models` sẽ dùng môi trường cách ly vì dry-run cho thấy thay đổi nhiều dependency lõi của `.venv`. |
| 26/09/2026 | 🔄 PhoWhisper cloud PTQ | AIMET local vẫn bị chặn trên Windows, nhưng Workbench hỗ trợ quantization phía cloud. Đã dùng checkpoint-pinned wrapper tạo PhoWhisper encoder PT2, chuyển sang ONNX `j5qld7vop` và quantize w8a16 `jpvln12r5` thành công. Smoke dùng một tensor random chỉ để kiểm tra compile/latency, không có giá trị WER. QNN compile/profile và calibration thật đang là bước kế tiếp; WSL không còn là hard blocker cho baseline cloud PTQ. |
| 26/09/2026 | ⚠️ PhoWhisper float experiment | Compile trung gian float16 QNN DLC `jpyo2oel5` thành công nhưng link context binary `j5ql1l8op` fail vì target không nhận `input_features` float. Không có artifact GPU deployable. File vendor `qai_hub_models/utils/args.py` từng bị sửa để bỏ check FP16 đã được khôi phục bằng reinstall package 0.63.0; không giữ patch trong môi trường. |
| 26/09/2026 | 🔄 PhoWhisper decoder PTQ/QNN | Decoder PT2 → ONNX `jp2rz716g` thành công. Hai job calibration sai đã dừng: `j5680q30g` dùng tên input cũ, `jp4y2jm8p` gửi dataset rỗng. Workbench helper đã sửa để đọc 51 tensor từ target ONNX input spec; job đúng `jp4y2j38p` đã quantize thành công một mẫu random 125.264.680 bytes thành QDQ ONNX `mmrg5yr6q`. QNN DLC compile `jpyoyv2l5` đang chạy. Random calibration không có giá trị WER. |
| 27/09/2026 | ⛔ PhoWhisper generic-PTQ blocker | Encoder/decoder QNN compile pass, nhưng direct profile `jgddmm3lg`/`j5wlrrl6p` và diagnostic link `jp0mxxm6g`/`jp8ekkexp` đều fail. Log QAIRT 2.50 xác định `node_matmul` cần DSP architecture `>=73`, QCS6490 là v68. Kiểm tra source package sửa lại một giả định: float recipe đã gọi `monkey_patch_model()` để MHA→SHA và linear→conv; khác biệt còn lại là quantized recipe dùng specialized AIMET per-tensor config, equalization và calibrated encodings. Custom PhoWhisper không được tái dùng encodings OpenAI vì weights khác; phải tạo QuantSim với `aimet_encodings=None` và calibrate lại trên Linux/WSL, hoặc chọn model nhỏ hơn. |
| 27/09/2026 | ⛔ Piper QNN graph blocker | Workbench helper đã hỗ trợ input specs JSON có validation. Piper EN/VI bucket 128 phoneme chuyển ONNX tĩnh và cloud PTQ random-smoke đều pass. Compile đầu `jgk2kk22g`/`j5qlddl4p` thiếu `--truncate_64bit_io`; retry đúng cờ `jp4y224vp`/`jpxlzzr1p` vẫn fail QAIRT exit 255. Log xác định cả hai graph có `NonZero` sinh shape động và node `/Range` phụ thuộc `/ReduceMax_output_0`, QNN không hỗ trợ dynamic value này. Dừng submit graph hiện tại; cần export/rewrite duration path để Range/NonZero có shape tĩnh, hoặc giữ CPU ARM fallback. |
| 27/09/2026 | ✅ Local QNN graph gate | Thêm `scripts/check_onnx_qnn_compat.py` để fail-fast trước upload khi ONNX có `NonZero` hoặc `Range` bị điều khiển bởi runtime `ReduceMax`. Cả Piper EN và VI có 13 blocker chắc chắn (12 `NonZero`, 1 `ReduceMax→Range`) và nhiều Range/Shape warning có thể được constant-fold sau static export. Report: `data/reports/aihub/tts/piper_qnn_static_shape_gate.json`. |
| 27/09/2026 | ✅ Linux AIMET handoff | `scripts/prepare_phowhisper_aimet.py` fail-closed ngoài Linux, không sửa `site-packages`, nhận checkpoint thật và chỉ lấy role=train deterministic. Encoder tạo specialized Qualcomm encodings từ audio thật. Decoder pipeline mới tạo cross/self KV-cache từ optimized float encoder/decoder và transcript teacher-forced, kiểm tra đúng thứ tự/tên/shape/dtype của 51 input, stream tensor trong RAM và chỉ lưu evidence hash/count không nhạy cảm. Contract và state progression có unit test; cần smoke Linux `limit=2, steps=2` trước khi chạy calibration đầy đủ hoặc upload. Lệnh handoff nằm trong `GPU_HANDOFF.md`. |
| 26/09/2026 | ⚠️ TTS Piper BYOM | Hai job `jpe7xd8v5` (EN) và `jgjryvoep` (VI) đều fail với `Input tensor input has dynamic shape [-1, -1], which is not supported`; không có job nào bị hủy. Piper cần compile với input specs tĩnh hoặc export graph fixed-length. CPU fallback mới chỉ đo trên PC; phải benchmark lại trên CPU ARM của board trước khi chốt deployment. |

## 1. Mục tiêu, phạm vi và tiêu chí thành công

OneVoice là bản thử nghiệm dịch giọng nói y tế hai chiều Việt ↔ Anh. Mục tiêu là giữ dữ liệu riêng tư và có thể chạy không cần mạng.

**Mục tiêu của toàn hệ thống:** người nói một ngôn ngữ, người nghe nhận được cả chữ và giọng nói ở ngôn ngữ còn lại.

**Lý do dùng nhiều bước:** ASR, MT và TTS có thể được đo và sửa riêng. Nếu một bước sai, hệ thống biết phải chặn hoặc yêu cầu người dùng xác nhận.

```text
microphone
→ VAD/khử nhiễu (tìm đoạn có lời và làm sạch audio)
→ ASR (voice → text cùng ngôn ngữ)
→ kiểm tra thuật ngữ
→ MT (text VN ↔ text EN)
→ kiểm tra an toàn
→ TTS (text → voice ngôn ngữ đích)
```

### 1.1 Mục tiêu bắt buộc

- ASR Việt (**voice VN → text VN**) chép được hội thoại y tế, câu trộn Việt–Anh và giọng Bắc/Trung/Nam.
- MT (**text VN ↔ text EN**) chỉ dịch; không tư vấn và không tự thêm chẩn đoán hoặc liều lượng.
- Giữ nguyên số, đơn vị, tên thuốc, phủ định và mức độ khẩn cấp.
- TTS (**text → voice**) phải đọc rõ; màn hình vẫn hiện câu gốc và câu dịch để người dùng kiểm tra.
- Có benchmark (mốc đo) cho model gốc, model INT8 và model trên thiết bị thật; mục tiêu chưa đo không được ghi thành kết quả.
- Dataset phải qua EDA (kiểm tra dữ liệu) trước merge (gộp); train/validation/test phải tách theo người nói hoặc lần ghi âm và phải có người nghe mẫu.

### 1.2 Không nằm trong phạm vi prototype hiện tại

- Không chẩn đoán, kê đơn hoặc thay bác sĩ.
- Không cam kết HIPAA/GDPR/thiết bị y tế nếu chưa qua quy trình pháp lý tương ứng.
- Không cam kết `<1.4s`, `MOS >4.25`, `WER <15%` hay thời lượng pin trước khi đo.
- Không train từ scratch; không distill bằng chain-of-thought.
- Không xây phần cứng 3 microphone/beamforming nếu chưa có board và thiết kế điện tử thực.

> **Mục tiêu đo sau này:** tổng thời gian phản hồi dưới 2 giây và RTF dưới 1. Đây là ngưỡng mong muốn, chưa phải kết quả trên QCS6490.

## 2. Ràng buộc dự án một người

| Ràng buộc | Quyết định vận hành |
|---|---|
| Một người phát triển | Tự động hóa các bước và dừng an toàn khi có lỗi. **Lý do:** giảm việc thủ công và tránh duy trì quá nhiều model cùng lúc. |
| GPU thuê online | Máy local chỉ kiểm tra dữ liệu/model; chuẩn bị đủ lệnh và file trước khi thuê GPU. **Lý do:** không trả tiền GPU trong lúc sửa lỗi cơ bản. |
| Ngân sách hữu hạn | Đo baseline trước, rồi mới fine-tune model phù hợp nhất. **Lý do:** chỉ chi tiền khi có mốc so sánh rõ. |
| QCS6490 khó chạy model lớn | Thử INT8 trước và luôn giữ đường chạy CPU dự phòng. **Lý do:** model phải chạy thật trên board, không chỉ chạy trên PC. |
| Dữ liệu y tế rủi ro cao | Không lưu audio người dùng mặc định; xóa hoặc che PII. **Lý do:** bảo vệ quyền riêng tư. |

> **Update:** mọi đoạn “team 4 người”, Thái/Khoa/Minh bị loại khỏi **nguồn sự thật hiện hành**. Hai DOCX proposal được giữ byte-for-byte như bản gốc theo yêu cầu chủ dự án; không chỉnh, không chèn báo cáo. File này là nơi duy nhất ghi trạng thái thực tế dự án một người.

## 3. Kiểm tra mã nguồn hiện tại

**Mục tiêu:** hiểu project đang có gì, phần nào sai và phần nào có thể dùng tiếp.

**Lý do:** sửa đúng lỗi gốc trước khi tốn thời gian cho dữ liệu và GPU.

### 3.1 Có sẵn

- Pipeline Python: nhận audio, ASR (**voice → text**), MT (**text VN ↔ text EN**), TTS (**text → voice**), cache câu nhanh và bộ điều phối.
- Script huấn luyện ASR/MT, thử INT8, đo chất lượng và tạo nhiễu khi train.
- Proposal song ngữ và nghiên cứu dữ liệu/model cũ.
- Notebook merge cũ để tham khảo ý tưởng nguồn dữ liệu.

### 3.2 Vấn đề đã phát hiện và cách xử lý

| Vấn đề cũ | Mức độ | Cách sửa và lý do |
|---|---:|---|
| Dùng model chỉ biết tiếng Anh để nhận giọng Việt | Rất cao | Đổi sang `PhoWhisper-small`, vì mục tiêu là **voice VN → text VN**. |
| Gộp dữ liệu rồi chia ngẫu nhiên | Rất cao | Giữ test/hard được khóa và chia theo người nói/nhóm ghi âm, vì cùng nội dung ở nhiều tập sẽ tạo điểm cao giả. |
| Cách băm cũ không thấy bản trùng giữa các nguồn | Cao | Băm nội dung chuẩn bằng SHA-256, vì tên file khác không có nghĩa audio khác. |
| Eka chỉ có test nhưng từng được định dùng để train | Rất cao | Giữ Eka chỉ để đo ASR tiếng Anh, vì dùng test để học sẽ làm mất tính khách quan. |
| Chat tiếng Việt một ngôn ngữ bị coi là cặp dịch | Rất cao | Loại khỏi MT, vì **text VN → text EN** cần cặp câu thật ở cả hai ngôn ngữ. |
| Code cũ ghép sai hai nửa MedEV | Rất cao | Ghép câu EN ở nửa đầu với câu VI cùng vị trí ở nửa sau, vì đây mới là cặp dịch đúng. |
| Dùng chat LLM cho dịch máy nhỏ gọn | Cao | Chọn NLLB encoder-decoder, vì mục tiêu chỉ là dịch ổn định chứ không tạo hội thoại. |
| Gọi UINT8 là INT4 | Cao | Bỏ claim sai; chỉ thử INT8 có đo chất lượng, vì tên kỹ thuật sai sẽ dẫn tới kế hoạch thiết bị sai. |
| Ghi cứng thời gian token đầu là 120 ms | Cao | Xóa số chưa đo, vì latency phải lấy từ lần chạy thật. |
| Chỉ có một điểm tổng | Trung bình | Thêm điểm theo vùng, vai trò, nhiễu và code-switch, vì điểm trung bình có thể che lỗi nguy hiểm. |
| Phiên bản thư viện chưa được khóa | Trung bình | Tách bộ thư viện local/GPU, vì máy thuê phải tái tạo được môi trường. |
| Audio trùng bit nhưng ID/video khác nhau | Rất cao | Băm từng FLAC, ưu tiên giữ test rồi validation rồi train, vì không được để cùng audio ở nhiều tập. |

> **Trạng thái:** code training/inference đã sửa đúng kiến trúc. Quantization scaffold hiện export PhoWhisper/NLLB seq2seq và chỉ tạo dynamic-QInt8 ONNX làm desktop parity candidate; không gọi đó là calibrated/QNN INT8. Scaffold fail-closed với INT4, thiếu SNPE SDK, converter lỗi hoặc không sinh output. Calibration, QNN model library/context binary và benchmark thiết bị chỉ chạy sau khi có checkpoint fine-tuned thật.

## 4. Model được chọn và mục tiêu của từng model

### 4.1 ASR tiếng Việt — voice VN → text VN

**Mục tiêu:** chép giọng nói tiếng Việt thành văn bản tiếng Việt, kể cả câu y tế và câu trộn từ tiếng Anh.

**Lý do chọn:** đây là đầu vào bắt buộc trước khi dịch VN → EN; sai ở bước này sẽ truyền lỗi sang MT và TTS.

**Model chính:** `vinai/PhoWhisper-small` → fine-tune trên VietMed train + ViMedCSS train. Model này bắt đầu từ Whisper đa ngôn ngữ và đã học trên 844 giờ tiếng Việt nhiều vùng giọng.

**Model dự phòng:** `vinai/PhoWhisper-base` nếu bản small quá chậm hoặc quá nặng trên board.

Không thêm hàng nghìn token y tế một cách cơ học. Tên thuốc/ICD được xử lý bằng data coverage, contextual biasing hoặc post-ASR terminology matcher; tokenizer expansion chỉ thử như ablation có benchmark.

**English:** giữ `distil-whisper/distil-small.en` làm baseline. Chưa fine-tune vì Eka là test-only; cần tìm train corpus độc lập, license rõ và speaker-disjoint.

> **Trạng thái:** training script đã có `--preflight`, yêu cầu local audio manifest, kiểm tra transcript/speaker/recording-group overlap **và SHA-256 audio trùng bit**, và dừng rõ nếu không có CUDA. Base model/processor được tải đúng commit `a86b604c…` đã khóa thay vì theo tên floating. Train collator đọc FLAC 16 kHz sau QC và augment mới mỗi batch bằng speed 0,90–1,10, gain ±4 dB, colored/hum noise SNR 12–35 dB, short reverb, telephone band-limit và μ-law codec; validation luôn giữ nguyên. Ablation local giữ greedy: beam-4 chỉ giảm WER validation từ 19,40% xuống 18,99% nhưng chậm khoảng 3 lần và làm xấu một số slice; prompt y khoa tĩnh bị loại vì từng đổi sai `Bartholin` thành `insulin`. Prompt theo ngữ cảnh chỉ được thử lại nếu người dùng chọn chuyên khoa rõ ràng và có safety review theo câu.

### 4.2 MT — text VN ↔ text EN

**Mục tiêu:** dịch văn bản hai chiều Việt ↔ Anh và giữ đúng số, đơn vị, thuốc, tên riêng và ý phủ định.

**Lý do chọn:** OneVoice là công cụ dịch, không phải chatbot; model dịch chuyên dụng dễ kiểm soát hơn model hội thoại.

**Model chính:** `facebook/nllb-200-distilled-600M`, fine-tune **một checkpoint dùng chung cho cả hai chiều** trên MedEV.

Lý do chọn NLLB:

- được thiết kế cho dịch văn bản, phù hợp hơn chatbot;
- đầu ra dễ kiểm soát hơn và ít tự thêm ý;
- có mã ngôn ngữ EN/VI rõ ràng và chạy ổn định;
- vẫn phải kiểm chứng size/latency và license CC-BY-NC-4.0.

**Candidate chất lượng:** `vinai/vinai-translate-{vi2en,en2vi}-v2`; chỉ dùng nếu AGPL-3.0 tương thích cách phát hành.

**Distillation:** tắt. Chỉ mở nếu baseline không đạt terminology/adequacy. Nếu mở, chỉ lưu source + final translation + QA metadata, không lưu chain-of-thought.

> **Trạng thái:** training script đã dùng `AutoModelForSeq2SeqLM`; không còn tạo dummy data khi dataset thiếu. Base tokenizer/model được tải đúng commit `f8d333a0…` đã khóa. Joint preflight pass trên 339.028 cặp gốc = 678.056 train examples hai chiều và 17.858 validation examples, exact overlap = 0. Full tokenizer EDA dùng context 256: train EN/VI vượt context 0,0726%/0,0923%; validation 0,1008%/0,1456%; test 0,0783%/0,1007%, đều dưới gate 0,20%. Validation trong train chọn checkpoint bằng loss; sinh câu/BLEU/chrF/safety chạy riêng vì forced target-language token khác nhau theo từng hướng. Decode runtime dùng greedy (`num_beams: 1`) theo kết quả validation: BLEU/chrF2 25,93/46,02, cao hơn beam-4 24,13/45,26 và nhanh hơn khoảng 4,8 lần; locked test cho thấy chênh BLEU nhỏ nhưng chrF2 greedy nhỉnh hơn, safety bằng nhau.

### 4.3 TTS — text VN/EN → voice VN/EN

**Mục tiêu:** đọc bản dịch thành giọng nói ở ngôn ngữ đích.

**Lý do chọn:** người dùng có thể nghe ngay, nhưng câu chữ vẫn được hiển thị để kiểm tra trước khi phát nội dung nhạy cảm.

- Việt: baseline thực thi `vi_VN-vivos-x_low`; `vi_VN-vais1000-medium` chỉ là candidate human AB sau.
- Anh: `en_US-lessac-medium`.
- Trước fine-tune TTS: tạo test suite 100–200 câu gồm thuốc, số, đơn vị, viết tắt, code-switch và câu khẩn cấp; human AB/MOS.
- TTS phải đọc dosage có cấu trúc; nếu safety checker phát hiện số/đơn vị thay đổi, không phát audio tự động.

> **Trạng thái/đo local:** đã tải đúng ONNX + config cho VI/EN, cache session và fail-closed nếu thiếu artifact. Smoke warm inference: VI 455,91 ms / 4,112 s audio (RTF 0,111); EN 280,34 ms / 4,447 s (RTF 0,063). Piper báo thiếu một số phoneme với tên thuốc ngoại; medical alias mặc định tắt và chỉ bật từng từ sau khi nghe xác minh. Hai WAV ở `data/reports/smoke/tts_{vi,en}.wav`.

## 5. Dữ liệu — kiểm tra trước, gộp sau

**Mục tiêu:** tạo dữ liệu sạch, đa dạng và không bị rò giữa train/validation/test.

**Lý do:** nhiều dữ liệu chưa chắc tốt; file sai, file trùng hoặc chia sai có thể làm model yếu và điểm đánh giá cao giả.

### 5.1 Registry nguồn lõi

| Dataset | Dùng cho mục tiêu nào | Số lượng quan sát | Độ đa dạng | Quyết định và lý do |
|---|---|---:|---|---|
| VietMed | ASR Việt: **voice VN y tế → text VN** | train 2.773; dev 2.912; test 3.437; cv 85 | vùng giọng, giới tính, bác sĩ/bệnh nhân/người dẫn, ICD-10, điều kiện thu | Dùng train/dev; khóa test. `cv` chỉ hiệu chỉnh vì trùng người nói với train. |
| ViMedCSS | ASR Việt có trộn tiếng Anh: **voice VN/EN → text đúng câu nói** | 11.832 / 1.714 / 1.614 / hard 658 | chủ đề, từ tiếng Anh, nhóm video | Dùng train/validation; khóa test/hard để đo câu khó. |
| MedEV | MT y tế: **text EN ↔ text VN** | 681.794 / 17.878 / 17.920 dòng thô | cặp câu Anh–Việt | Ghép hai nửa đúng vị trí và giữ split gốc, vì ghép sai sẽ dạy model dịch sai. |
| Eka Medical ASR EN | Đo ASR Anh: **voice EN → text EN** | test 3.619 | phiên thu, khái niệm, câu và thực thể | Chỉ dùng test vì nguồn không có train hợp lệ. |
| VIVOS | Bổ sung ASR Việt phổ thông: **voice VN → text VN** | train 11.660; calibration 760 | 65 người nói, phòng yên tĩnh | Giới hạn 80 câu/người để dữ liệu phổ thông không lấn át dữ liệu y tế. |
| ViMD | Ứng viên đo giọng 63 tỉnh | khoảng 19.000 / 102,56 giờ | Bắc/Trung/Nam, người nói, giới tính | Chưa dùng vì tải lớn và giấy phép CC BY-NC-ND cần xem xét thêm. |
| Bud500 | Ứng viên bổ sung ASR Việt | bị giới hạn truy cập, khoảng 500 giờ | Bắc/Trung/Nam | Chưa cần ở vòng đầu; chỉ lấy mẫu nếu dữ liệu hiện tại thiếu vùng giọng. |
| VietSpeech | Ứng viên bổ sung ASR Việt đời thường | bị giới hạn truy cập, khoảng 1.100 giờ | vùng giọng, cách nói và phong cách | Chưa dùng vì quá lớn cho vòng đầu của dự án một người. |
| FPTU Vovinam | Ứng viên ASR Việt phổ thông | 14.048/1.756/1.756 | tỉnh, người nói, môi trường; train chủ yếu miền Trung | Không gộp vì 96,8% dòng train lặp câu và người nói bị rò giữa các split. |
| ViVoice34 preview | Ứng viên theo tỉnh | 381 dòng xem trước | 22 tỉnh, 57 loại câu | Không gộp vì nguồn ẩn danh, clip quá dài và gói đầy đủ gần 29,9 GB. |
| GovVox-100h-v3 | Ứng viên đo vùng giọng | cần xin quyền truy cập | người nói, tỉnh, giới tính, vùng | Không dùng khi chưa được cấp quyền và giấy phép chưa rõ. |
| FutureBeeAI | Ứng viên MT: **text VN ↔ text EN** | API trả 401 | chưa xác minh | Không phụ thuộc vào nguồn chưa truy cập được. |
| Vietnamese medical chat | Hỏi đáp y tế chỉ có tiếng Việt | 46.479 | câu hỏi–trả lời y tế | Không dùng cho MT vì không có câu dịch tiếng Anh tương ứng. |

### 5.2 EDA — kiểm tra từng nguồn trước khi dùng

**Mục tiêu:** biết rõ dữ liệu có bao nhiêu, có nghe được không, có trùng không và có đúng giấy phép không.

**Lý do:** chỉ gộp sau khi hiểu từng nguồn; nếu gộp trước, lỗi sẽ khó tìm và có thể lan sang toàn bộ dataset.

- cấu trúc dữ liệu, số dòng, ô thiếu, độ dài audio và tổng số giờ;
- độ dài câu, số ký tự mỗi giây và audio không khớp transcript;
- bản trùng trong cùng split và giữa các split;
- người nói, phiên thu, recording hoặc video xuất hiện ở nhiều split;
- nguồn, lĩnh vực, vùng giọng, giới tính, vai trò, nhóm bệnh, điều kiện thu và câu trộn ngôn ngữ;
- ký tự lỗi, chuẩn Unicode và transcript sai ngôn ngữ;
- giấy phép, quyền truy cập và quyền chia sẻ lại;
- mẫu ngẫu nhiên lẫn mẫu rủi ro để nghe thật.

Artifact: `data/reports/eda/<dataset>.json`, metadata JSONL theo split và `REPORT.md`.

> **Kết quả full EDA 22/09/2026:**
>
> - **VietMed:** 9.207 clips, 15,929 giờ. Train 4,799h (South East 1.406, North 1.246, Central Highland 121); dev 4,958h; test 6,022h; cv 0,149h. Không transcript chéo split. `cv` trùng 13 speaker/5 recording với train nên chỉ calibration. Có 1 recording (`VietMed_019`) chéo dev/test; merge sẽ giữ test và loại phần validation xung đột.
> - **ViMedCSS:** 15.818 clips, 32,643 giờ. Train 24,304h; validation 3,568h; test 3,389h; hard 1,383h. Có 102 transcript và 1.413 YouTube video chéo role; riêng video overlap: test↔train 1.045, test↔validation 601, train↔validation 909. Không có metadata accent/speaker; bắt buộc group-disjoint merge. `hard` có transcript English-only và được báo slice riêng.
> - **MedEV:** 717.592 raw rows ghép đúng thành 358.796 cặp (340.897/8.939/8.960). Không lỗi hàng lẻ; overlap chính xác là 41 pair train↔validation, 48 train↔test và 0 validation↔test. Audit alignment sâu cho median tỷ lệ ký tự VI/EN 0,977–0,980. `number_set_mismatch` (~15%) chủ yếu là dấu thập phân/ngày tháng/thông tin hợp lệ nên chỉ là review slice; sai ngôn ngữ/identity, tỷ lệ độ dài <0,25 hoặc >4 và cặp >2.000 ký tự là fatal. License upstream chưa khai báo; owner chỉ cho research/non-commercial/no-redistribution.
> - **Eka EN:** 3.619 clips test-only, 8,401 giờ; narration entity 2.206, narration sentence 1.303, conversation 110. Có 292 transcript lặp (nhiều entity ngắn), không phải bằng chứng audio trùng; giữ benchmark và báo slice/type. Một outlier text/audio ratio rất cao phải nghe trong pack.
> - **VIVOS:** full metadata EDA pass trên 11.660 train + 760 calibration; 46 train speakers và 19 calibration speakers, 0 speaker/group leakage. Có 178 transcript train↔calibration và 1.327 duplicate-text rows trong train do scripted multi-speaker; không coi đó là audio duplicate. Upstream row metadata không có duration nên duration/signal được xác minh lúc materialize. Chỉ lấy tối đa 80 câu mỗi train speaker; owner đã nghe pack 12 mẫu ≥2 giây tại `data/review/vivos_listening_pack/` và xác nhận audio–transcript khớp 100%.

### 5.3 Merge gate — điều kiện để được gộp dữ liệu

**Mục tiêu:** chỉ đưa dữ liệu đã đạt kiểm tra vào train/validation/test cuối.

**Lý do:** đây là cửa chặn cuối để lỗi của một nguồn không làm bẩn toàn bộ dữ liệu.

Merge chỉ chạy khi:

1. `full_audit=true`;
2. status là `pass` hoặc `reviewed` có ghi lý do;
3. license được duyệt cho mục đích cuộc thi/nghiên cứu;
4. không speaker/recording leakage ngoài thiết kế benchmark;
5. locked test fingerprint được nạp trước train/validation;
6. fatal flags bị loại khỏi mọi role để cả train và benchmark đều sạch; cờ mềm được giữ và gắn slice.

Ưu tiên ownership duplicate: medical curated > medical code-switch > general. Test không bao giờ bị xóa để ưu tiên train.

> **Kết quả merge 22/09/2026:** split gốc ViMedCSS không group-disjoint; nếu bảo toàn tuyệt đối chỉ còn 2.167 ViMedCSS train clips/4,516h. Vì vậy giữ hai bộ artifact:
>
> - `manifests_strict_official`: 4.940 ASR train clips/9,315h để so sánh theo split nguồn sau khi loại video leakage.
> - `manifests` primary trước signal-QC: khóa toàn bộ `hard` cùng video liên quan vào test; phần còn lại group-hash 85/7,5/7,5. Sau signal-QC **và exact-audio SHA-256 dedup**: 17.806 train/31,860h (2.773 VietMed + 11.354 ViMedCSS + 3.679 VIVOS), 3.733 validation/6,619h (2.764 VietMed + 969 ViMedCSS) và 10.390 test/20,912h (3.436 VietMed + 3.346 ViMedCSS + 3.608 Eka).
> - Primary validation: 0 duplicate ID, 0 speaker overlap, 0 recording-group overlap và 0 exact-audio hash trùng trong/chéo role. Có 55 transcript giống nhau chéo role được giữ vì audio hash khác và chỉ báo thành prompt-overlap slice.
> - Final signal EDA: duration train p01/p50/p99 = 2/6/15 giây (min 0,875 giây vẫn nghe hợp lệ); RMS train p01/p50 = 0,01355/0,07499; decoded CPS p99=26,25 và max=34,5, đều dưới fatal gate 35. Clipped fraction lớn nhất 5,90%, dưới gate 20%. Hai clip DC cao đã được sửa thay vì xóa. 74 train/3 validation/51 test hàng `no_vietnamese_marks` được giữ làm missing-diacritic/English/code-switch review slice, không coi thiếu dấu tự động là lỗi transcript.
> - MT: 339.028/8.929/8.938 pairs train/validation/test; loại 854 fatal-quality, 953 exact duplicate nội role, 92 pair overlap với role ưu tiên cao hơn và 2 contact-PII; exact-pair overlap cuối = 0. SHA-256 được ghi tại `data/reports/manifests/validation.json`.

### 5.4 Human listening QA — người thật nghe và đối chiếu

**Mục tiêu:** xác nhận audio nghe được và transcript đúng với lời nói.

**Lý do:** máy có thể đo âm lượng hoặc độ dài nhưng không hiểu chắc câu nói có khớp hay không.

Mỗi nguồn nghe tối thiểu 12 mẫu, gồm:

- 3 random theo source/split;
- 3 outlier characters/second hoặc duration;
- 2 accent/role hiếm;
- 2 code-switch/thuốc khó;
- 2 duplicate/leakage/English-only đáng ngờ.

Mỗi mẫu ghi: `audio_ok`, `transcript_exact`, `language`, `accent_if_confident`, `pii`, `keep/drop`, `note`. Không đoán accent khi không chắc. Tỷ lệ sai transcript >5% hoặc PII chưa xử lý sẽ block nguồn.

> **Trạng thái:** đã tái tạo **36 clips** (12/source: VietMed, ViMedCSS, Eka) với phân bố duration cân bằng. Toàn bộ dài ≥2 giây và RMS gốc ≥0,003; hai mẫu 0,48 giây/1,00 giây thực sự rác/khó nghe đã bị loại, ghi vào `signal_exclusions.jsonl`, được thay bằng mẫu đạt gate, và merge script loại đúng các ID này khỏi manifest. Đây là quyết định theo chất lượng tín hiệu, **không phải quy tắc xóa mọi audio dưới 2 giây**. Nghe qua `data/review/listening_pack/review.html`; playback là mono 16 kHz PCM16 đã chuẩn hóa peak, audio gốc nằm trong `audio/`. Chủ dự án xác nhận audio và transcript 36/36 khớp. MedEV đã được owner cho phép research-only/no-redistribution. Pack bổ sung VIVOS cũng đã được owner xác nhận 12/12 khớp; tổng human QA hiện là 48/48 mẫu keep, không phát hiện PII.

### 5.5 Augmentation — tạo biến thể chỉ cho tập train

**Mục tiêu:** giúp ASR (**voice → text**) chịu được tốc độ nói, âm lượng, nhiễu, vang phòng, điện thoại và codec khác nhau.

**Lý do:** môi trường thật đa dạng hơn dữ liệu sạch; validation/test phải giữ nguyên để phép đo vẫn công bằng.

- Chỉ augment **train**; tuyệt đối không augment validation/test.
- Vòng GPU đầu dùng noise tổng hợp không vướng license: white/pink/50–60 Hz hum, SNR liên tục 12–35 dB; không phá lời bằng SNR quá thấp trước khi có ablation.
- Không nhân mọi clip 3× cố định; dùng on-the-fly probability và giữ clean sample.
- Có speed perturbation nhẹ, gain, short reverb, μ-law codec và telephone 300–3.400 Hz; mọi output clip về `[-1,1]`.
- Noise/RIR thực chỉ thêm sau khi có nguồn license rõ và ablation chứng minh không làm giảm clean/medical WER.
- Báo WER clean, noise theo SNR và telephone riêng; “75 dB môi trường” không đồng nghĩa một SNR duy nhất.

## 6. Cách đo chất lượng

**Mục tiêu:** đo riêng từng model và toàn hệ thống bằng số liệu có thể lặp lại.

**Lý do:** một điểm tổng không cho biết lỗi nằm ở ASR, MT, TTS hay thiết bị.

### 6.1 Đo ASR — voice → text cùng ngôn ngữ

**Mục tiêu:** biết model chép sai bao nhiêu từ, ký tự, số, đơn vị và thuật ngữ y tế.

**Lý do:** câu dịch không thể đúng nếu transcript đầu vào sai.

- Chỉ số chính là WER (tỷ lệ sai từ); chỉ số phụ là CER (tỷ lệ sai ký tự), lỗi thuật ngữ y tế và độ chính xác của số/đơn vị.
- Report raw + normalized text; normalizer versioned và dùng giống nhau cho mọi model.
- Slice: dataset, accent, doctor/patient, gender, recording condition, ICD group, code-switch, SNR.
- 95% bootstrap confidence interval; không chọn model chỉ vì chênh < interval.
- Compare: base float → fine-tuned float → INT8 ONNX/QNN → on-device.

### 6.2 Đo MT — text VN ↔ text EN

**Mục tiêu:** kiểm tra câu dịch đúng nghĩa và không làm đổi thông tin y tế quan trọng.

**Lý do:** BLEU cao vẫn có thể nguy hiểm nếu sai liều, đơn vị hoặc phủ định.

- SacreBLEU signature đầy đủ, chrF++, COMET (nếu compute cho phép).
- Chỉ số an toàn chính: đúng thuật ngữ, giữ số/đơn vị, không đảo ý phủ định, giữ tên riêng và không tự thêm/bỏ ý.
- Đánh giá EN→VI và VI→EN riêng; MedEV test khóa.
- Human review tối thiểu 100 câu stratified theo emergency, medication, procedure, symptom và long sentence.

### 6.3 Đo TTS — text → voice

**Mục tiêu:** giọng đọc rõ, đúng thuốc, số, đơn vị và viết tắt.

**Lý do:** bản dịch đúng chữ nhưng đọc sai vẫn có thể gây hiểu nhầm.

- MOS/AB human, pronunciation error rate cho thuốc và viết tắt, number/unit intelligibility.
- RTF và latency trên CPU/device; cold/warm riêng.
- Không dùng MOS target làm số đo giả.

### 6.4 An toàn toàn tuyến — voice nguồn → voice đích

**Mục tiêu:** so sánh câu gốc, transcript ASR và bản dịch MT trước khi TTS phát tiếng.

**Lý do:** nếu số, đơn vị, thuốc hoặc phủ định thay đổi thì phải yêu cầu xác nhận thay vì tự đọc.

- Source vs ASR vs MT: số/đơn vị, polarity, medication/entity diff.
- Nếu ASR confidence thấp hoặc guard fail: hiển thị “cần xác nhận”, không tự phát TTS.
- Emergency flash phrase chỉ kích hoạt bằng exact/controlled intent, có confirmation cho câu có hành động nguy hiểm.

> **Trạng thái:** locked CPU baseline đã chạy xong: PhoWhisper VI WER 20,87%/CER 15,39% (24 mẫu), Distil-Whisper EN WER 25,54%/CER 7,85% (24 mẫu), NLLB joint BLEU 23,76/chrF2 44,96 (24 câu, 0/24 safety failure sau sửa guard). Đây là base-model baseline trước fine-tune, không phải kết quả QCS6490.

### 6.5 Smoke test CPU — chạy thử để biết pipeline hoạt động

**Mục tiêu:** xác nhận model gốc tải được và mỗi bước có đầu ra thật trên máy local.

**Lý do:** smoke test chỉ tìm lỗi vận hành; số đo ít mẫu không đại diện cho chất lượng cuối.

- Smoke cũ chạy **từng model riêng**, không phải full cascade: PhoWhisper-small trên Piper VI còn sai nhiều (`tìm`, tên thuốc), latency 9.705 ms CPU; điều này xác nhận cần real-audio benchmark/fine-tune, không dùng confidence proxy 0,945 làm bảo đảm.
- Distil-Whisper EN trên Piper EN: câu dosage gần đúng hoàn toàn, latency 4.986 ms CPU.
- NLLB VI→EN 12.273 ms, EN→VI 2.307 ms CPU; cả hai giữ số `0,5/0.5`, `mg` và phủ định trong smoke guard.
- Full artifact: `data/reports/smoke/cpu_smoke.json`. Các số phụ thuộc máy và cold/warm cache, chưa đại diện QCS6490.

### 6.6 Full base cascade — audio thật → ASR → MT → safety → TTS

**Mục tiêu:** kiểm tra các model base có nối được thành một lượt hoàn chỉnh và tạo được audio đầu ra ở cả hai chiều.

**Lý do:** từng model chạy riêng chưa chứng minh output của bước trước tương thích với bước sau.

Đã chạy trên hai clip thật trong locked test:

- VI→EN: VietMed 6,0 giây → ASR WER 17,86% → MT tạo câu EN → Piper tạo 6,304 giây audio. Tổng CPU warm 23,276 giây, RTF 3,879.
- EN→VI: Eka 4,68 giây → ASR WER 20,00% → MT tạo câu VI → Piper tạo 2,336 giây audio. Tổng CPU warm 4,226 giây, RTF 0,903.
- Cả hai lượt đều đi qua safety guard và tạo WAV, nên **operational smoke = pass**.
- Chất lượng chưa đạt: `post prandial` bị ASR nghe thành `post cranial`, sau đó MT dịch sai thành “sau khi sọ”. Safety guard hiện chỉ bắt số/đơn vị/phủ định/thuật ngữ đã biết nên không phát hiện được lỗi nghĩa này. **Lý do cần fine-tune và human evaluation:** lỗi ASR có thể truyền qua toàn chuỗi dù code không crash.
- Report: `data/reports/smoke/base_e2e/report.json`; audio: `vi_to_en.wav`, `en_to_vi.wav` cùng thư mục.

Giới hạn phải ghi rõ:

- Đây là 2-sample smoke, không dùng p50/p95 của hai mẫu làm benchmark.
- Input là audio đã cắt sẵn; chưa test microphone và VAD chia đoạn trực tiếp.
- Hai tập ASR không có câu dịch đích nên không tính BLEU end-to-end; chất lượng MT riêng đã đo trên 24 cặp MedEV.
- Ở smoke lịch sử này, Silero VAD không tải vì Torch Hub yêu cầu trust tương tác nên hệ thống cũ dùng energy fallback; `noisereduce` cũng chưa cài nên denoise pass-through. Từ hardening 08/10, runtime đã thay đường đó bằng WebRTC VAD local-only đã smoke thật, không còn tải mạng/energy fallback. Noise suppression hiện được khai báo explicit `enabled=false` và vẫn phải có backend + clinical validation riêng trước live demo; nó không còn giả vờ đã chạy khi thực tế pass-through.

Kế hoạch test sau fine-tune:

1. Chạy lại đúng hai clip smoke để phát hiện lỗi tích hợp.
2. Chạy 24 clip thật mỗi chiều để đo ASR, safety failure, latency và tỷ lệ tạo audio thành công.
3. Chạy tập song ngữ có câu dịch chuẩn để đo MT end-to-end; không tự gán reference cho dữ liệu không có.
4. Chủ dự án nghe các WAV có thuốc, số, đơn vị và câu phủ định.
5. Cuối cùng mới test microphone + VAD + thiết bị QCS6490 và báo p50/p95/RTF thật.

## 7. Cách pipeline chạy khi sử dụng

**Mục tiêu:** xử lý một lượt nói hoàn chỉnh theo chuỗi **voice → text → bản dịch → voice**.

**Lý do:** Whisper hiện cần đủ ngữ cảnh; cắt thành từng mảnh 500 ms độc lập sẽ làm câu chữ kém chính xác.

Whisper chưa chạy streaming thật theo từng mảnh 500 ms. Thiết kế phù hợp hơn là:

1. VAD tìm đoạn có lời theo khung 20–30 ms và giữ một vùng audio gần nhất;
2. ASR tạo text tạm trên cửa sổ 4–15 giây;
3. chỉ chốt phần text đã ổn định;
4. MT dịch theo cụm hoặc câu đã chốt, không dịch vài từ rời;
5. TTS chỉ đọc câu đã qua kiểm tra an toàn.

Latency phải tách:

- capture/end-of-speech latency;
- ASR first partial/final;
- MT full sequence;
- TTS time-to-first-audio;
- end-to-end p50/p95, cold/warm.

> **Trạng thái:** prototype hiện là cascaded batch-like pipeline; chưa được phép gọi là streaming production.

> **Safety hardening:** flash cache mặc định chỉ nhận exact normalized phrase, không substring/fuzzy. Câu cache chứa hành động điều trị/thủ thuật được trả về để hiển thị nhưng đặt `requires_confirmation=true` và không tự phát TTS; custom cache phrase mặc định cũng yêu cầu xác nhận trừ khi được review rõ ràng. File-mode CLI chỉ tổng hợp/phát câu loại này sau khi người vận hành nhập chính xác `PLAY`; audio sink tính lại safety từ source/translation ngay trước dispatch nên sửa text hoặc metadata sau inference bị chặn. ASR và MT vẫn là stage bắt buộc; riêng lỗi vận hành TTS sau một bản dịch đã pass safety có thể hạ cấp theo cấu hình sang text-only, kèm `degraded_mode`/`degradation_code` ổn định và tuyệt đối không phát audio lỗi.

## 8. Đưa model lên Qualcomm QCS6490

**Mục tiêu:** chạy ASR, MT và TTS trên thiết bị thật với tốc độ, bộ nhớ và nhiệt độ chấp nhận được.

**Lý do:** model chạy trên GPU/PC chưa chắc biên dịch hoặc chạy tốt trên chip QCS6490.

### 8.1 Điều đã biết hiện tại

- Qualcomm AI Hub có recipes Whisper/Distil-Whisper, nhưng asset/quantized path cho QCS6490 phải xác minh theo model cụ thể.
- QCS6490 HTP v68 thực tế ưu tiên INT8; compilation cần calibration và quantized I/O.
- Llama/Qwen lớn có rủi ro memory/runtime; NLLB cũng phải thử compile, không mặc định hỗ trợ.
- QNN output có thể là context binary, không nên đồng nhất với `.dlc` của SNPE.

### 8.2 Thứ tự triển khai và lý do

1. Chạy model float bằng PyTorch để xác nhận đầu ra đúng; đây là mốc gốc.
2. Xuất ONNX float và so đầu ra; bước này kiểm tra việc chuyển định dạng không làm đổi model.
3. Tạo INT8 bằng dữ liệu hiệu chỉnh đại diện; bước này giảm kích thước/tăng tốc nhưng phải giữ chất lượng.
4. Biên dịch bằng QNN cho QCS6490 và lưu log; bước này xác nhận chip hỗ trợ graph.
5. Chạy trên board 15–30 phút để đo latency, RAM, nhiệt và điện; bước này kiểm tra điều kiện thật.
6. Nếu graph không chạy, chia một phần sang CPU hoặc chọn model nhỏ hơn; mục tiêu là có hệ thống chạy được thay vì giữ model quá nặng.

Acceptance: quantized metric degradation ≤ ngưỡng đã đặt, không unsupported op, không OOM, p95 và thermal ổn định.

> **Trạng thái:** chưa có checkpoint fine-tuned và chưa có board artifact; mọi claim kích thước/latency cũ bị hạ thành hypothesis.

## 9. Bảo mật, riêng tư và giấy phép

**Mục tiêu:** không làm lộ dữ liệu y tế và không dùng nguồn/model ngoài phạm vi giấy phép.

**Lý do:** chất lượng kỹ thuật không đủ nếu project vi phạm quyền riêng tư hoặc quyền sử dụng.

- Không commit audio, model weights, token hoặc PII; `.gitignore` phải bao phủ cache/report nhạy cảm.
- Hash không phải anonymization; transcript có PII phải redaction hoặc loại bỏ.
- Log runtime mặc định không chứa raw audio/full transcript.
- Ghi dataset revision/commit, checksum, license snapshot và ngày tải.
- MedEV dataset card chưa khai báo license rõ: dùng research prototype có ghi nguồn, block commercial use cho đến khi xác nhận.
- ViMedCSS có nhãn CC-BY-4.0 nhưng audio từ YouTube: cần review redistribution trước upload derivative dataset.
- NLLB CC-BY-NC-4.0 và VinAI Translate AGPL-3.0 ảnh hưởng phát hành sản phẩm.

> **Trạng thái:** đã tạo `DATA_LICENSES.md`. MedEV và NLLB hiện bị giới hạn research/non-commercial; ViMedCSS không được redistribute audio YouTube nếu chưa review quyền.

## 10. Kế hoạch triển khai cho một người

Mỗi phase có một mục tiêu rõ để tránh làm nhiều việc cùng lúc.

### Phase A — Hiểu và sửa nền tảng

**Mục tiêu:** làm cho code và giả định ban đầu đúng. **Lý do:** các bước dữ liệu/train phía sau phụ thuộc vào nền tảng này.

- [x] Audit repo/notebook cũ.
- [x] Sửa model/data assumptions nguy hiểm.
- [x] Thêm dataset registry và quality primitives.
- [x] Thêm EDA API không cần tải toàn bộ audio.
- [x] Thêm merge gate, audio materializer và unit tests.
- [x] Tách dependency local/GPU và thêm readiness preflight.

### Phase B — Kiểm tra dữ liệu trước khi gộp

**Mục tiêu:** hiểu và duyệt từng nguồn. **Lý do:** không đưa dữ liệu chưa rõ chất lượng hoặc giấy phép vào model.

- [x] Smoke EDA mọi nguồn lõi.
- [x] Full EDA VietMed/ViMedCSS/MedEV/Eka.
- [x] Review leakage định lượng và thêm group/speaker-aware merge repair.
- [x] Listening pack 36 clips; chủ dự án xác nhận transcript khớp 100%.
- [x] Human gate bổ sung VIVOS: owner xác nhận 12/12 audio–transcript khớp 100%; review artifact đã ghi nhận.
- [x] MedEV research-only/no-redistribution và mọi audit đã `reviewed`.
- [x] PII candidate scan + reviewed exclusion registry.
- [x] Full MedEV tokenizer-length EDA trên train/validation/test; context 256 và truncation gate pass.

### Phase C — Gộp dữ liệu và tải audio thật

**Mục tiêu:** tạo manifest và audio local sạch. **Lý do:** training phải đọc đúng file đã được kiểm tra và khóa hash.

- [x] Merge manifest primary + strict-official; validation checksum/leakage pass.
- [x] Tải audio bằng URL refresh/resume; 32.055/32.055 assets có mặt trước signal-QC.
- [x] Tạo `DATASET_CARD.md`, source/model revision lock và SHA-256 manifest reports.
- [x] Chuẩn hóa lossless mono 16 kHz FLAC; signal-QC pass, final paths portable và missing=0.
- [x] Băm toàn bộ FLAC sau QC; loại 46 bản sao chính xác, cross-role exact-audio overlap=0 và khóa inventory 31.929 file.

### Phase D — Chạy thử local và tạo mốc gốc

**Mục tiêu:** chứng minh pipeline hoạt động trước khi thuê GPU. **Lý do:** sửa lỗi trên CPU rẻ hơn sửa trong giờ GPU tính phí.

- [x] Đã tải base model/processor và CPU smoke inference thật.
- [x] Baseline khóa: 24 VI ASR + 24 EN ASR + 24 MT hai chiều; prediction/report/slices/CI đã lưu.
- [x] Tải Piper voices và tạo WAV pronunciation smoke.
- [x] Training preflight ASR/MT pass; hỗ trợ BF16/FP16, 100-step dry-run, data limit và resume checkpoint.
- [x] Cloud GPU handoff hoàn tất; mặc định budget A40/RTX A6000 48 GB, có profile 24 GB và đo throughput trước rồi mới chốt chi phí.
- [x] Ablation decoding local trên locked validation: giữ greedy cho ASR/MT; loại beam search và prompt y khoa tĩnh theo tiêu chí chất lượng, safety và tốc độ.
- [x] Error mining mẫu khó: ASR nổi bật ở thuật ngữ hiếm/tên bệnh (`Bartholin`, `Bell`, `corticosteron`), audio điện thoại/sách và đảo/mất cụm; MT nổi bật ở tim mạch, viết tắt và thuật ngữ ghép. Đây là tín hiệu định hướng sampling/challenge set, chưa được coi là thống kê tổng thể vì mới 24 mẫu.
- [x] Validation stratified đã khóa riêng ở `data/eval/mt_selection_dev.jsonl` (256 cặp) và `data/eval/asr_selection_dev.jsonl` (384 câu), chỉ sinh từ validation; bake-off preflight kiểm tra lại disjoint train/test trước khi dùng.
- [x] Terminology challenge set có 128 lỗi validation đã xác minh ở `data/eval/terminology_challenge_set.jsonl`: 110 lỗi MT safety và 18 lỗi ASR thật từ nhiều run. Mọi hàng có `hazard_level`, `review_status=pending`, `evaluation_only=true` và `auto_correction_eligible=false`; report provenance/hash nằm cạnh artifact và tuyệt đối không dùng hàng chưa duyệt để tự sửa output.

### Phase E — GPU fine-tune (điểm dừng hiện tại)

**Mục tiêu:** huấn luyện và so sánh ASR/MT trên selection-dev, rồi chỉ mở blind test cho winner vượt hard safety gate. **Lý do:** một training run hoàn tất không đủ chứng minh an toàn y tế hoặc đủ điều kiện promotion.

- [x] Fine-tune PhoWhisper-small Candidate A full-data; locked evaluation fail clinical gate nên chỉ giữ làm reference.
- [x] Fine-tune NLLB-600M Candidate A full-data; locked evaluation fail clinical gate nên chỉ giữ làm reference.
- [ ] Hoàn tất challenger bake-off: M2M100 VI→EN r32 phải resume từ checkpoint 100 đã xác minh; sau đó chạy Whisper-small multilingual rồi PhoWhisper-base tuần tự. Các profile MT hoàn tất trước đó đều fail clinical gate và không được promotion.
- [ ] Chọn winner theo locked selection-dev, hard safety và 95% CI; chỉ mở blind v2 một lần khi dữ liệu unseen sẵn sàng. Chưa có winner hợp lệ.

### Phase F — Tối ưu và đưa lên thiết bị sau GPU

**Mục tiêu:** giảm model xuống INT8, chạy trên QCS6490 và đo toàn tuyến. **Lý do:** chỉ tối ưu model sau khi đã có checkpoint fine-tuned đạt chất lượng.

- [ ] Quantization parity và calibration.
- [ ] QNN compile/deploy QCS6490.
- [ ] End-to-end safety/latency/power/thermal benchmark.
- [ ] Demo UI + fallback behavior + final evidence pack.

## 11. Các lệnh để chạy lại

**Mục tiêu:** một người vẫn có thể tái tạo từng bước theo đúng thứ tự.

**Lý do:** tránh phụ thuộc vào trí nhớ và giảm nguy cơ chạy nhầm dữ liệu.

```powershell
# Local environment
.venv\Scripts\python.exe -m pytest -q

# EDA đầy đủ trước merge
.venv\Scripts\python.exe scripts\audit_datasets.py --output-dir data\reports\eda
.venv\Scripts\python.exe scripts\audit_mt_token_lengths.py

# Human listening QA (đã tạo pack 36 clips)
.venv\Scripts\python.exe scripts\build_listening_pack.py --per-source 12
.venv\Scripts\python.exe scripts\apply_listening_review.py --reviewer "<ten-cua-ban>"

# Sau khi audit được pass/reviewed
.venv\Scripts\python.exe scripts\merge_manifests.py --allow-reviewed --policy-mode strict --output-dir data\processed\manifests_strict_official
.venv\Scripts\python.exe scripts\merge_manifests.py --allow-reviewed --policy-mode configured
.venv\Scripts\python.exe scripts\validate_manifests.py
.venv\Scripts\python.exe scripts\materialize_audio.py --workers 12
.venv\Scripts\python.exe scripts\qc_local_audio.py --workers 8
.venv\Scripts\python.exe scripts\summarize_local_audio.py
.venv\Scripts\python.exe scripts\build_audio_inventory.py --workers 8

# CPU preflight trước thuê GPU
.venv\Scripts\python.exe -m src.training.finetune_whisper_vi --preflight
.venv\Scripts\python.exe -m src.training.finetune_mt_medical --direction joint --preflight
.venv\Scripts\python.exe scripts\run_baseline_benchmarks.py --task asr --language vi --samples 24
.venv\Scripts\python.exe scripts\run_baseline_benchmarks.py --task asr --language en --samples 24
.venv\Scripts\python.exe scripts\run_baseline_benchmarks.py --task mt --samples 24
.venv\Scripts\python.exe scripts\run_base_e2e_pipeline.py
.venv\Scripts\python.exe scripts\build_selection_dev.py
.venv\Scripts\python.exe scripts\build_terminology_challenge_set.py
.venv\Scripts\python.exe scripts\preflight_project.py --gate gpu --output data\reports\preflight_gpu_ready.json
```

GPU command chỉ được chạy sau khi manifest checksum và preflight artifact đã khóa.

## 12. Checklist bàn giao sang GPU thuê

**Mục tiêu:** bật GPU là có thể kiểm tra rồi chạy dry-run ngay.

**Lý do:** thời gian GPU có tính phí; dữ liệu, môi trường, resume và ngân sách phải sẵn sàng trước.

- [x] Source/model revisions và manifest SHA-256 khóa trong `configs/artifact_lock.yaml` + reports.
- [x] `requirements-gpu.txt` pin dependency; CUDA PyTorch lấy từ image/provider rồi verify bằng lệnh trong handoff.
- [x] Dataset path portable, clean FLAC + local manifest không chứa Windows absolute path.
- [x] Inventory 31.929 FLAC duy nhất có size + SHA-256; cloud preflight xác minh lại toàn bộ sau upload.
- [x] Trainer hỗ trợ checkpoint/resume; output phải ở persistent volume.
- [x] Khuyến nghị budget A40/RTX A6000 48 GB FP16; profile 24 GB có batch/accumulation riêng; bắt buộc 100-step measured dry-run trước full run.
- [x] Logging loss/throughput/VRAM; raw medical data không đưa lên public tracker.
- [x] Budget alert USD 10 cho hai dry-run đầu + terminate compute ngay sau sync; ngân sách full run chỉ chốt sau số đo throughput thật.
- [x] Lệnh dry-run/full/resume và acceptance nằm trong `GPU_HANDOFF.md`.

> **Điểm báo chủ dự án đã đạt:** Phase B–D hoàn tất. Bắt đầu bằng A40/RTX A6000 48 GB và persistent volume ≥150 GB; profile 24 GB chỉ dùng sau dry-run xác nhận VRAM/throughput. Không đoán tổng giờ: chạy 100 step cho từng model, đo throughput rồi mới chốt estimated hours/cost. A100 80 GB chỉ là fast path tùy chọn.

## 13. Quyết định còn mở

**Mục tiêu:** ghi rõ việc nào đã chốt và việc nào phải chờ bằng chứng mới.

**Lý do:** không tự thay đổi hướng model hoặc dữ liệu mà thiếu lý do.

| ID | Quyết định | Trạng thái | Default an toàn |
|---|---|---|---|
| D-01 | Có dùng MedEV khi giấy phép nguồn chưa ghi rõ không? | Chủ dự án đã duyệt 22/09/2026 | Chỉ nghiên cứu/phi thương mại, không chia sẻ lại; thay nguồn nếu làm sản phẩm thương mại. |
| D-02 | Các dòng chỉ có tiếng Anh trong tập `hard` của ViMedCSS được đo thế nào? | Đã chốt | Giữ trong test khó và báo riêng nhóm tiếng Anh/câu trộn ngôn ngữ. |
| D-03 | Có thêm nguồn phổ thông/vùng giọng không? | Đã chốt sau EDA mở rộng | Dùng VIVOS cân bằng theo người nói và nhãn vùng của VietMed. Không gộp FPTU Vovinam/ViVoice34/GovVox vì rò dữ liệu, nguồn gốc hoặc quyền truy cập chưa đạt. |
| D-04 | MT dùng NLLB, VinAI hay M2M100? | Mở lại có kiểm soát 05/10/2026 | Hoàn tất NLLB Candidate A; sau đó bake-off theo chiều. VinAI AGPL bị khóa trước GPU tới khi có phê duyệt research-license; M2M100-418M MIT là fallback production. |
| D-05 | ASR dùng PhoWhisper-small, Whisper-small hay PhoWhisper-base? | Bake-off sau Candidate A | So sánh successive halving; safety hard gate và 95% CI trước blind v2, rồi mới đo winner trên board. |

## 14. Điều kiện để được thuê và chạy GPU

**Mục tiêu:** chỉ trả tiền GPU khi toàn bộ dữ liệu và code local đã sẵn sàng.

**Lý do:** preflight chặn lỗi sớm và bảo đảm dry-run đo đúng pipeline sẽ dùng để fine-tune.

Chỉ đánh dấu ready khi tất cả điều sau đúng:

- Full EDA artifacts tồn tại và được review.
- Human listening QA 48/48 hoàn tất (36 core + 12 VIVOS), quyết định keep/drop có log.
- Không còn train/validation/test leakage chưa giải quyết, gồm speaker/group/text policy và exact-audio SHA-256.
- Local audio đầy đủ, duration/header/checksum pass; inventory không có hash trùng và khớp artifact lock.
- ASR/MT train + validation manifests có số lượng cố định và SHA-256.
- Base model CPU smoke inference pass.
- Training preflight pass và 20–100 step GPU dry-run config sẵn sàng.
- Output storage/resume/budget/auto-shutdown rõ ràng.

**Hiện tại: đã sẵn sàng cho GPU. Các bước local bắt buộc đã đạt; có thể tiếp tục chạy validation stratified và xây terminology challenge set trên CPU để tăng độ chắc chắn. Tuy nhiên, bước có khả năng cải thiện trọng số model rõ rệt vẫn là thuê GPU và chạy thử fine-tune 100 bước theo `GPU_HANDOFF.md`; không nên CPU-fine-tune hai model 600M/Whisper-small chỉ để tránh thuê GPU.**
