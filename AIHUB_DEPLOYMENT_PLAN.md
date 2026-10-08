# OneVoice — Qualcomm AI Hub deployment plan

> Trạng thái 27/09/2026: Qualcomm AI Hub CLI `0.55.0` và AI Hub Models CLI `0.63.0` đã được cài trong `.venv`. Tài khoản đã nhìn thấy `Dragonwing RB3 Gen 2 Vision Kit`, device OS `Qc_Linux 1.6`, chipset `qualcomm-qcs6490`. QAIRT khả dụng: `2.45`, `2.49`, `2.50` (`default`, `latest`). Artifact tham chiếu Whisper Small w8a16 đã profile; PhoWhisper encoder/decoder random-calibration smoke compile QNN DLC pass nhưng direct profile fail `MODEL_GRAPH_ERROR`; xem mục 5A.
>
> Mục tiêu của kế hoạch này là chứng minh kiến trúc OneVoice chạy được trên QCS6490 **trước khi** trả tiền fine-tune. AI Hub không thay GPU training; nó dùng để compile, quantize, profile, chạy inference và tải artifact triển khai.

## 1. Kết quả cuối cần đạt

OneVoice chỉ chuyển sang triển khai board khi có đủ bằng chứng sau:

1. ASR, MT và TTS có artifact deployable hoặc có quyết định fallback rõ ràng.
2. Mỗi artifact có compile log, profile latency/memory/compute-unit và kiểm tra numerical parity.
3. INT8 không làm hỏng số, đơn vị, phủ định, tên thuốc hoặc thuật ngữ đã khóa.
4. Ước lượng model-only cho thấy toàn pipeline còn khả năng đạt mục tiêu tổng dưới 2 giây và RTF dưới 1.
5. Checkpoint fine-tuned chỉ được tạo sau khi kiến trúc base tương ứng vượt gate triển khai.
6. Acceptance cuối được đo lại trên board vật lý; số từ AI Hub không được dùng như kết quả end-to-end cuối.

## 2. Phạm vi và nguyên tắc

### Trong phạm vi

- Qualcomm AI Hub Workbench, AI Hub Models và QNN/QAIRT.
- Target `Dragonwing RB3 Gen 2 Vision Kit`, `Qc_Linux 1.6`, QCS6490.
- PhoWhisper-small, NLLB-200 distilled 600M và Piper VI/EN.
- INT8 calibration, QNN context binary, profile và inference parity.
- CPU fallback khi NPU không mang lại lợi ích hoặc graph không được hỗ trợ.

### Ngoài phạm vi ở vòng đầu

- Không distill trước khi có kết quả INT8 thật.
- Không fine-tune trên AI Hub; fine-tune dùng GPU thuê riêng.
- Không upload toàn bộ dataset hoặc audio có PII.
- Không gọi latency AI Hub là latency sản phẩm hoàn chỉnh.
- Không đổi model chỉ vì một compile job lỗi; phải đọc log và thử đường export đúng trước.

## 3. Hợp đồng đo độ trễ

Trước khi so sánh model, dùng ba loại số đo riêng:

| Chỉ số | Bắt đầu → kết thúc | Nơi đo | Mục đích |
|---|---|---|---|
| Model latency | Tensor input → tensor output của một graph | AI Hub profile | Chọn graph/runtime/precision. |
| Response latency | Frame có tiếng cuối → sample audio đầu tiên | Board/app | Mục tiêu trải nghiệm dưới 2 giây. |
| Full-processing RTF | Tổng thời gian xử lý / thời lượng audio đầu vào | Board/app | Phải nhỏ hơn 1. |

Ngân sách lập kế hoạch ban đầu, chưa phải kết quả đo:

| Thành phần | Ngân sách tạm |
|---|---:|
| Endpoint/VAD sau frame có tiếng cuối | 400 ms |
| Audio frontend + chuyển tensor | 50 ms |
| ASR encoder + decoder | 650 ms |
| MT encoder + decoder | 450 ms |
| Safety/cache/orchestration | 50 ms |
| TTS đến sample audio đầu tiên | 350 ms |
| Overhead còn lại | 50 ms |
| Tổng | 2.000 ms |

`silence_duration_ms` hiện là 800 ms, vì vậy measurement trên board phải quyết định có giảm endpoint delay hay không. Không giảm trước khi test tỷ lệ cắt mất cuối câu.

## 4. Quy ước bảo mật và artifact

### Credential

- [ ] Xác nhận token từng xuất hiện trong chat đã bị revoke.
- [x] Không lưu API token trong repository.
- [x] Dùng cấu hình người dùng của `qai-hub`, không truyền token trong script hoặc log.
- [ ] Trước khi chia sẻ log, scan và loại token, URL ký tạm thời và dữ liệu nhạy cảm. Backup schema 2 nay tự quét source và đọc lại tar theo streaming, không echo secret và fail-closed. Generator mới không còn persist URL; downloader refresh URL từ đúng datasets-server trong RAM, bắt public HTTPS rồi loại trước khi ghi manifest. Dry-run migration xác minh 25/33 artifact EDA local còn 42.106 trường `audio_url`; gate vẫn mở cho tới khi owner chủ động chạy sanitizer `--apply` hoặc tái tạo artifact.

### Dữ liệu được phép upload

- Calibration/inference chỉ dùng tensor đã tiền xử lý từ dữ liệu public hoặc đã xác nhận không có PII.
- Ưu tiên ID mẫu và tensor; không upload transcript/audio raw không cần thiết.
- Calibration ASR lấy mẫu stratified theo nguồn, accent, role, recording condition và code-switch.
- Calibration MT lấy câu theo hai hướng, độ dài và nhóm số–đơn vị–phủ định.

### Cấu trúc đầu ra dự kiến

```text
data/reports/aihub/
  environment.json
  jobs.jsonl
  asr/
    base_float_reference.json
    calibration_manifest.jsonl
    compile.json
    profile.json
    parity.json
  mt/
    base_float_reference.json
    calibration_manifest.jsonl
    compile.json
    profile.json
    parity.json
  tts/
    compile.json
    profile.json
    listening_review.csv

models/aihub/
  asr/
  mt/
  tts/
```

`models/aihub/` là artifact lớn và không commit. Report chỉ lưu metadata, metric, job ID và checksum; không lưu credential.

## 5. Phase 0 — Khóa môi trường

### Việc làm

- [x] Xác nhận CLI:

```powershell
& ".\.venv\Scripts\qai-hub.exe" --help
```

- [x] Xác nhận target:

```powershell
& ".\.venv\Scripts\qai-hub.exe" list-devices |
    Select-String "QCS6490|RB3"
```

- [x] Xác nhận framework:

```powershell
& ".\.venv\Scripts\qai-hub.exe" list-frameworks
```

- [x] Ghi package version, QAIRT version, OS target và thời điểm vào `data/reports/aihub/environment.json`.
- [x] Dùng QAIRT `2.50` làm baseline vì hiện là `default`, `latest` và khớp asset Whisper QCS6490; chỉ A/B với bản cũ nếu có regression cụ thể.

### Gate 0

- Device xuất hiện đúng QCS6490/Qc_Linux 1.6.
- CLI truy cập được account mà không truyền token trên command line.
- Không còn token bị lộ đang hoạt động.

### 5A. Kết quả thực thi thật ngày 26/09/2026

- Đã tải `Whisper-Small-Quantized` QNN context binary w8a16 chính thức cho QCS6490; encoder 119.214.520 bytes, decoder 223.713.576 bytes, checksum được khóa trong report tham chiếu.
- Catalog QCS6490 báo encoder `233,30 ms`, decoder `23,46 ms/token`, đều chạy NPU. Ước lượng encoder + 20 token là `702,50 ms`, đã vượt ngân sách ASR tạm `650 ms`; 40 token là `1.171,70 ms`.
- Encoder profile job `jp2ro876g` (có truyền options) báo lỗi nội bộ. Job A/B `jgoldyjxg` bỏ toàn bộ profile option **đã thành công** trên thiết bị thật: 100 lần đo cho p50 `239,419 ms`, p95 `240,818 ms`; peak memory `36,23 MB`, 1.894 operation đều trên NPU. Điều này chứng minh Whisper-small reference load toàn bộ trên NPU. A/B cho thấy options là khác biệt chính nhưng chưa đủ để khẳng định riêng `--compute_unit` là nguyên nhân duy nhất của lỗi server.
- Decoder profile job `jpyo8ev05` **thành công** trên thiết bị thật: 100 lần đo cho min `22,954 ms`, p50 `34,089 ms`, p95 `37,197 ms`, max `45,403 ms` mỗi token; 2.853 operation đều trên NPU, estimated peak `74.653.696 bytes`. Catalog `23,46 ms/token` gần với min nhưng không đại diện p50/p95 sustained.
- `PiperTTS-EN` chính thức không có download/performance cho QCS6490. Piper VI không có trong catalog. TTS vì vậy là BYOM. Hai job ONNX gốc `jpe7xd8v5` (EN) và `jgjryvoep` (VI) đều thất bại với lỗi `dynamic shape [-1, -1]`; không có job nào bị hủy. QAIRT cần input specs tĩnh trước khi compile. Vì CPU local hiện đã đạt RTF `0,0591` (VI) và `0,0772` (EN), CPU trên board vẫn là fallback dự kiến nhưng phải benchmark lại trên ARM board.
- Catalog chỉ có OpusMT EN↔ES và EN↔ZH, không có VI. NLLB EN↔VI bắt buộc BYOM và không được coi là deployable trước khi encoder/decoder/cache đều profile thành công.
- Full `qai-hub-models` đã được cài vào `.venv-aihub-models` cách ly, không thay dependency của `.venv`. AIMET-ONNX local vẫn chỉ hỗ trợ Linux/WSL, nhưng **không còn là hard blocker**: AI Hub Workbench có quantize job chạy AIMET phía cloud từ ONNX + calibration tensors. Encoder PhoWhisper đã chuyển PT2 → ONNX thành công (`j5qld7vop`) và cloud PTQ w8a16 smoke thành công (`jpvln12r5`). Calibration smoke dùng một tensor random nên chỉ chứng minh đường kỹ thuật, tuyệt đối chưa chứng minh WER/parity.
- Encoder QNN compile `jgzl0x045` và decoder QNN compile `jpyoyv2l5` pass, nhưng direct profile `jgddmm3lg`/`j5wlrrl6p` và diagnostic link `jp0mxxm6g`/`jp8ekkexp` đều fail; `node_matmul` cần DSP architecture `>=73`, QCS6490 là v68. Source audit xác nhận float recipe đã chạy `monkey_patch_model()` để MHA→SHA và linear→conv; vì vậy blocker không chỉ là thiếu graph rewrite. Official quantized recipe còn dùng specialized AIMET config, cross-layer equalization và calibrated per-tensor encodings. Muốn giữ PhoWhisper weights phải tạo QuantSim từ chính checkpoint (`aimet_encodings=None`) rồi compute encodings mới trên calibration train data; không được tái dùng official OpenAI encodings. Bước này cần Linux/WSL, nhưng không tự cài hệ thống.
- Piper EN/VI bucket tĩnh 128 phoneme đã chuyển ONNX và cloud PTQ thành công. QNN compile đầu `jgk2kk22g`/`j5qlddl4p` thiếu `--truncate_64bit_io`; retry đúng cờ `jp4y224vp`/`jpxlzzr1p` vẫn fail QAIRT exit 255. Log xác định `NonZero` tạo output shape động và `/Range` phụ thuộc dynamic `/ReduceMax_output_0`, không được QNN hỗ trợ. Dừng submit graph hiện tại; bước tiếp theo là rewrite/export duration path có shape tĩnh hoặc xác nhận CPU ARM fallback.
- Local pre-submit gate `scripts/check_onnx_qnn_compat.py` xác nhận mỗi Piper graph có 13 blocker chắc chắn: 12 `NonZero` data-dependent và một `ReduceMax→Range`; report nằm ở `data/reports/aihub/tts/piper_qnn_static_shape_gate.json`. Không submit lại cho đến khi graph mới qua gate này.
- Linux handoff `scripts/prepare_phowhisper_aimet.py` đã sẵn sàng cho cả hai component: nhận checkpoint local, kiểm tra kiến trúc, chọn deterministic train-only calibration và tạo specialized QuantSim/encodings mà không sửa package. Decoder lấy token thật từ transcript train, chạy optimized float encoder/decoder để tạo cross/self KV-cache theo teacher forcing, rồi kiểm tra đúng 51 input name/shape/dtype trước AIMET. Audio, transcript, token IDs và cache tensor chỉ tồn tại trong RAM; evidence chỉ lưu hash ID/count. Script dừng ngay trên Windows; decoder mới được kiểm tra bằng unit-test giả lập, còn phải smoke thật trên Linux trước mọi upload AI Hub. Encoder-only vẫn không phải artifact deployable.
- Thử nghiệm float trước đó chỉ tạo QNN DLC trung gian (`jpyo2oel5`) rồi fail khi link context binary (`j5ql1l8op`) vì `input_features` float không được target chấp nhận. Không được gọi đây là artifact GPU deployable. Bản vá tạm bỏ kiểm tra FP16 trong `site-packages` đã được gỡ bằng cách cài lại package Qualcomm nguyên bản.

## 6. Phase 1 — Smoke nền tảng và model Qualcomm

### 1A. Kiểm tra catalog

Cài CLI catalog nhẹ, không cài toàn bộ model dependencies trước khi cần:

```powershell
& ".\.venv\Scripts\python.exe" -m pip install qai-hub-models-cli
& ".\.venv\Scripts\qai-hub-models.exe" info Whisper-Small-Quantized
& ".\.venv\Scripts\qai-hub-models.exe" find whisper_small_quantized --chipset qualcomm-qcs6490 --all
```

Nếu tên executable khác sau khi cài, tìm bằng:

```powershell
Get-ChildItem .venv\Scripts -Filter "*hub*models*"
```

### 1B. Hai smoke job

1. [x] Xác nhận quyền upload/submit/profile trên target bằng Whisper Small Quantized chính thức.
2. [x] Tải và kiểm tra metadata/checksum asset Whisper Small Quantized QCS6490.
3. [x] Có ít nhất một profile job hoàn tất (Cả encoder và decoder đều đã thành công; job encoder bỏ option để tránh bug internal của server).

Không đoán `input_specs`. Lấy tên, dtype và shape trực tiếp từ model recipe/exported model trước khi submit.

### Gate 1

- Compile job thành công.
- Profile job chạy trên đúng `Dragonwing RB3 Gen 2 Vision Kit` và OS `1.6`.
- Có target artifact tải được.
- Có latency, load time, peak memory và compute-unit mapping.
- Inference output hữu hạn, đúng shape và không toàn NaN/zero.
- Không có full CPU fallback không giải thích được.

Nếu Gate 1 fail, dừng model OneVoice và xử lý theo thứ tự: input specs → quantized I/O → QAIRT version → unsupported op → model size.

## 7. Phase 2 — ASR feasibility trước fine-tune

### 2A. Hoàn thiện export graph

`src/training/quantize_qnn.py` hiện chỉ là desktop dynamic-QInt8 prototype và chưa đủ cho QCS6490. Cần bổ sung đường export riêng cho AI Hub:

- Whisper encoder.
- Decoder khởi tạo cache.
- Decoder dùng KV cache cho token tiếp theo.
- Fixed shapes cho batch 1, audio tối đa 30 giây và các decoder lengths cần thiết.
- Manifest ghi input/output name, dtype, shape và checksum từng graph.

Không dùng encoder-only success để kết luận ASR deployable.

### 2B. Float parity local

- Chạy cùng 24 mẫu validation bằng PyTorch và ONNX float.
- So text, logits/tokens và WER.
- Chỉ chuyển sang INT8 nếu ONNX float không có regression bất thường.

### 2C. Calibration INT8

- Chọn 500–1.000 audio từ **train**, không lấy test để calibration.
- Phân tầng theo VietMed/ViMedCSS/VIVOS, accent, role, Tel/Consultation/Book và code-switch.
- Tiền xử lý thành mel tensor local.
- Lưu ID/hash và lý do chọn; không lưu bản sao audio trong report.
- Quantize weights/activations và I/O theo yêu cầu QCS6490.
- Baseline thực hiện bằng `submit_quantize_job()` trên Workbench để Windows chỉ upload ONNX và calibration tensors. AIMET local/WSL chỉ là phương án nâng cao nếu cloud PTQ không đủ kiểm soát.

### 2D. Compile/profile/inference

- Compile từng graph cho QNN context binary.
- Profile encoder và mỗi decoder variant.
- Chạy inference 24 mẫu validation.
- Ước lượng ASR latency theo:

```text
encoder latency + decoder-init latency + N × decoder-with-cache latency
```

Trong đó `N` lấy từ phân bố token thật, báo p50 và p95; không chỉ dùng một câu ngắn.

### Gate 2

- Toàn bộ graph cần thiết compile và chạy được.
- Không có graph chính rơi toàn bộ xuống CPU mà không được chấp nhận rõ.
- WER INT8 tăng không quá 2 điểm phần trăm tuyệt đối so với cùng checkpoint float.
- Không tăng lỗi tên thuốc/số/đơn vị trong challenge slice.
- ASR p95 ước lượng không vượt ngân sách 650 ms, hoặc có kế hoạch giảm model/streaming cụ thể.

Nếu fail latency/OOM sau khi export đúng, thử Whisper-base/tiny trước khi fine-tune PhoWhisper-small.

## 8. Phase 3 — TTS feasibility

### Thứ tự

1. Piper EN theo recipe `PiperTTS-EN` của Qualcomm.
2. Piper VI ONNX hiện tại bằng BYOM.
3. So sánh CPU, QNN và nếu có TFLite/QNN delegate.

### Đo

- Load time và peak memory.
- Time-to-first-audio nếu runtime hỗ trợ streaming; nếu không, full synthesis latency.
- RTF theo câu ngắn/trung bình/dài.
- 20 câu thuốc, số, đơn vị, viết tắt và code-switch để nghe kiểm tra.

### Gate 3

- Không có lỗi phát âm mới do quantization/runtime.
- RTF nhỏ hơn 1.
- TTS-to-first-audio nằm trong ngân sách 350 ms, hoặc CPU hiện tại nhanh hơn và được chọn làm fallback chính thức.

Không ép Piper lên NPU nếu chi phí copy tensor hoặc giới hạn graph khiến nó chậm hơn CPU.

## 9. Phase 4 — MT feasibility trước fine-tune

### 4A. Export đúng kiến trúc sinh token

- NLLB encoder.
- Decoder-init.
- Decoder-with-cache.
- Forced target-language token cho `vie_Latn` và `eng_Latn`.
- Fixed source lengths đại diện, ví dụ 64/128/256.
- Decoder cache shapes cho batch 1.

### 4B. Float parity

- 24 cặp validation hai chiều.
- So PyTorch với ONNX float bằng BLEU, chrF2 và safety guard.
- Chặn pipeline nếu target-language token hoặc cache làm đổi hướng dịch.

### 4C. INT8 calibration và profile

- Chọn 500–1.000 cặp từ train, cân bằng hai hướng và độ dài.
- Bắt buộc có số, đơn vị, phủ định, viết tắt và thuật ngữ tim mạch/thuốc.
- Compile/profile encoder và decoder riêng.
- Ước lượng tổng thời gian theo số token đầu ra p50/p95.

### Gate 4

- BLEU hoặc chrF2 không giảm quá 1 điểm trên cùng validation slice.
- Safety failure không tăng; không chấp nhận thay số, đơn vị, phủ định hoặc mã y khoa.
- Không có token loop, NaN, sai language direction hoặc cache corruption.
- MT p95 ước lượng nằm trong ngân sách 450 ms.
- Tổng memory của graph/weights/cache phù hợp target.

Nếu NLLB không vượt Gate 4, dừng fine-tune NLLB. Chuyển sang khảo sát model encoder-decoder nhỏ hơn rồi lặp lại Phase 4; không distill một kiến trúc chưa chứng minh deployable.

## 10. Phase 5 — Quyết định trước GPU

Lập decision matrix:

| Model | Compile | Accuracy parity | p50/p95 | Peak memory | NPU coverage | Quyết định |
|---|---|---|---|---|---|---|
| PhoWhisper-small | Generic ONNX + cloud PTQ compile pass nhưng profile/link fail: `node_matmul` cần Hexagon ≥v73, QCS6490 chỉ v68 | CPU baseline WER 20,87%; random calibration không hợp lệ cho accuracy | Không đo được | Chưa đo | Không deployable với graph hiện tại | Dừng retry; port weights sang optimized quantized recipe hoặc chọn graph/model nhỏ hơn trước GPU. |
| Whisper Small w8a16 reference | Asset có sẵn; encoder & decoder profile pass | Chưa có QCS6490 numerics | Encoder thật p50 239,42 ms; decoder thật p50 34,09 ms/token | Encoder 36,23 MB, Decoder 74,65 MB | 100% NPU cho cả hai | 20 token ≈921 ms; không đạt ngân sách 650 ms, nhưng deployable. |
| Piper VI | Static-128 ONNX + cloud PTQ pass; QNN fail vì dynamic `NonZero → ReduceMax → Range` | Human QA 12/12; random PTQ chưa có parity | CPU synth 252,55 ms | Chưa đo board | CPU local | Rewrite/export duration path tĩnh hoặc giữ CPU ARM fallback; không retry graph hiện tại. |
| Piper EN | Static-128 ONNX + cloud PTQ pass; QNN fail vì dynamic `NonZero → ReduceMax → Range`; catalog không hỗ trợ QCS6490 | CPU smoke pass; random PTQ chưa có parity | CPU synth 342,23 ms | Chưa đo board | CPU local | Rewrite/export duration path tĩnh hoặc giữ CPU ARM fallback; không retry graph hiện tại. |
| NLLB-600M | Chưa BYOM | CPU BLEU 23,76/chrF2 44,96 | CPU 2,69–7,67 s/câu smoke | Chưa đo | CPU | Rủi ro cao; cần split encoder/decoder/cache trước GPU. |

Chỉ thuê GPU cho model có quyết định `fine-tune current architecture`.

Các quyết định hợp lệ:

- Fine-tune kiến trúc hiện tại.
- Đổi sang model nhỏ hơn rồi kiểm tra lại.
- Giữ CPU fallback.
- Loại khỏi prototype.

## 11. Phase 6 — Fine-tune tiết kiệm

### GPU

- Mặc định: A40 hoặc RTX A6000 48 GB, FP16.
- Dry-run ASR và MT riêng, mỗi job 100 steps.
- Batch train/eval 4, gradient accumulation 8, effective batch 32.
- Budget alert USD 10 cho hai dry-run đầu.
- Không chạy ASR và MT cùng lúc.

### Gate 6

- Không OOM.
- Loss có xu hướng giảm và không NaN.
- Throughput đủ để ước lượng full cost.
- Checkpoint/resume và persistent sync hoạt động.
- Chỉ chạy full sau khi chủ dự án chấp nhận dự toán từ throughput thật.

Distillation vẫn tắt.

## 12. Phase 7 — Fine-tuned INT8 regression

Lặp lại đúng Phase 2/4 với checkpoint fine-tuned:

```text
fine-tuned PyTorch
→ ONNX float parity
→ calibrated INT8
→ QNN context binary
→ QCS6490 profile
→ inference validation
→ locked test một lần sau khi chọn checkpoint
```

So sánh bốn cột:

1. Base float.
2. Fine-tuned float.
3. Fine-tuned INT8 trên AI Hub.
4. Fine-tuned INT8 trên board thật.

Không chấp nhận model chỉ nhanh hơn nhưng safety hoặc thuật ngữ xấu đi.

Mỗi winner phải sinh hai prediction JSONL trên **cùng manifest khóa**: float
reference và compiled INT8/QNN. Chạy `scripts/seal_quantization_parity.py` để
tái tính metric thay vì tin report tổng hợp: sealer kiểm toàn bộ metadata đầu
vào ngoài `hypothesis` giống hệt, khóa checksum manifest/predictions/artifact,
dùng paired bootstrap 1.000 lần và chỉ pass khi cả point estimate lẫn cận trên
95% CI có relative degradation không quá 2%. ASR bắt buộc ít nhất 32
group/speaker độc lập. Cả float và compiled output phải có zero failure trên
drug name, dose, number, unit, negation, terminology và code-switch. Deployment
finalizer từ chối winner thiếu parity artifact bất biến này.

## 13. Phase 8 — Board integration và acceptance cuối

### Tích hợp

- QAIRT/QNN runtime đúng version với artifact.
- Audio input, VAD, tokenizer, decoder loop, safety guard và playback trong cùng app.
- Cache exact-only; câu hành động lâm sàng vẫn yêu cầu xác nhận.
- CPU fallback có log rõ, không fallback im lặng.

### Benchmark

- Cold start và warm start.
- p50/p95 response latency.
- RTF theo audio 2/6/15/30 giây.
- RAM, peak memory, nhiệt và điện.
- Chạy liên tục 15–30 phút để đo throttling.
- Test hai chiều, code-switch, Tel, accent và câu có số/đơn vị/phủ định.

### Acceptance

- Response latency dưới 2.000 ms theo định nghĩa tại mục 3.
- RTF dưới 1.
- Flash-cache lookup dưới 50 ms.
- Không OOM/NaN/token loop.
- Safety không regression.
- Kết quả human review đạt trước demo.

## 14. Thứ tự thực thi ngắn gọn

1. Revoke token từng bị lộ và khóa environment report.
2. Catalog + generic compile/profile smoke.
3. Whisper Small Quantized reference smoke.
4. PhoWhisper base export/parity/INT8/profile.
5. Piper EN rồi Piper VI.
6. NLLB base export/parity/INT8/profile.
7. Chốt decision matrix.
8. Thuê GPU chỉ cho kiến trúc đã pass.
9. Fine-tuned INT8 regression trên AI Hub.
10. Board integration và acceptance cuối.

## 15. Điểm dừng và báo chủ dự án

Dừng và báo trước khi tiếp tục khi xảy ra một trong các điều sau:

- Cần upload dữ liệu có PII hoặc dữ liệu không được phép chia sẻ.
- NLLB/PhoWhisper cần thay kiến trúc, tokenizer hoặc context length.
- Cần trả phí GPU full run.
- INT8 vượt ngưỡng regression.
- Hosted profile đạt nhưng board thật không đạt do power/scheduling/thermal.
- Cần distillation hoặc model mới.

## 16. Tài liệu chính thức

- Qualcomm AI Hub Workbench: <https://dev.aihub.qualcomm.com/docs/>
- Getting started: <https://aihub.qualcomm.com/get-started>
- FAQ, BYOM, device profile, privacy và phí: <https://dev.aihub.qualcomm.com/docs/hub/faq.html>
- Quantization: <https://dev.aihub.qualcomm.com/docs/hub/quantize_examples.html>
- Qualcomm AI Hub Models: <https://github.com/qualcomm/ai-hub-models>
- Qualcomm AI Hub Apps: <https://github.com/qualcomm/ai-hub-apps>
