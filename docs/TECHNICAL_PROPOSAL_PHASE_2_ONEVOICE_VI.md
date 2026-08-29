# ĐỀ XUẤT KỸ THUẬT: PHASE 2 — TECHNICAL SUBMISSION
## CUỘC THI ONEVOICE AI CHALLENGE 2026

---

| Trường Thông Tin | Chi Tiết Dự Án |
| :--- | :--- |
| **Tên Đội thi / Dự án** | **MediVoice Edge** — Giải Pháp Dịch Thuật Y Tế Giọng Nói Siêu Độ Trễ, 100% On-Device Trên Qualcomm Hexagon NPU |
| **Ngày nộp hồ sơ** | 21 / 08 / 2026 (Nộp Đặc tả Kỹ thuật Phase 2) |
| **Phiên bản** | v1.0 (Bản Đề xuất Kỹ thuật Toàn diện) |
| **Mức độ bảo mật** | Bảo mật — Chỉ phục vụ đánh giá OneVoice AI Challenge |

---

## 1. Tóm Tắt Điều Hành (Executive Summary)

### 1.1 Tổng quan bài toán (Problem Overview)
Giao tiếp bằng giọng nói xuyên biên giới trong phòng cấp cứu (ER), phòng mổ và khu vực cách ly y tế đang đối mặt với rào cản vận hành nghiêm trọng. Các công cụ dịch thuật hiện nay hoàn toàn bất khả thi trong môi trường y tế do phụ thuộc vào máy chủ đám mây (Cloud) — vốn bị tê liệt ở tầng hầm bệnh viện hay phòng chụp chiếu bị cô lập sóng; gây ra độ trễ cao nguy hiểm (>4–6 giây); vi phạm quy định bảo mật dữ liệu sức khỏe bệnh nhân (HIPAA/GDPR cấm truyền âm thanh ra Cloud); và thường xuyên dịch sai các thuật ngữ y khoa cũng như hiện tượng chèn tiếng Anh (Code-switching) của bác sĩ.

### 1.2 Giải pháp đề xuất (Proposed Solution)
**MediVoice Edge** là thiết bị dịch thuật giọng nói hai chiều chuyên dụng dạng đeo ngực (wearable badge), hoạt động 100% độc lập On-Device (Zero Internet) trên nền tảng vi xử lý Qualcomm Snapdragon QCS6490 (RB3 Gen 2). Hệ thống tận dụng tối đa sức mạnh của lõi Hexagon NPU để mang lại độ trễ dịch thuật dưới 1.4 giây (Real-Time Factor RTF < 0.55) cho các cuộc hội thoại lâm sàng Tiếng Việt ↔ Tiếng Anh.

### 1.3 Đề xuất giá trị cốt lõi (Key Value Proposition)
1. **Bảo mật Tuyệt đối & 0% Truyền Dữ liệu Ra Ngoài (Air-Gapped Privacy):** 100% xử lý tại chỗ, đảm bảo tuân thủ đạo luật bảo mật thông tin y tế và miễn nhiễm hoàn toàn với sự cố mất mạng Internet.
2. **Độ trễ Siêu thấp Dưới 1.4s (RTF < 0.55):** Cơ chế chia nhỏ âm thanh streaming (audio chunking), lượng tử hóa ASR và suy luận dự đoán (speculative decoding) tối ưu cho phân loại cấp cứu khẩn cấp.
3. **Độ chính xác Thuật ngữ Y khoa & Code-Switching cao:** Ứng dụng kỹ thuật chưng cất tri thức (Knowledge Distillation) trên tập dữ liệu VietMed, ViMedCSS, MedEV và FutureBeeAI; dịch chuẩn xác danh mục bệnh ICD-10 và hiện tượng chèn thuật ngữ tiếng Anh của bác sĩ Việt Nam.
4. **Thiết kế Rảnh tay Kháng khuẩn chuẩn IP54:** Thiết bị dạng thẻ đeo ngực siêu nhẹ (<150g), tích hợp mảng 3 micro lọc ồn định hướng (beamforming) trong môi trường phòng bệnh ồn ào >75 dB và pin hoạt động >10 giờ liên tục.

---

## 2. Định Nghĩa Bài Toán & Người Dùng Mục Tiêu
*(Trọng số Đánh giá: 15% — Problem Definition & Impact)*

### 2.1 Tuyên bố bài toán (Problem Statement)
Trong phòng cấp cứu, phòng hồi sức tích cực (ICU) và các đoàn cứu trợ y tế quốc tế, tốc độ và sự chuẩn xác trong giao tiếp trực tiếp định đoạt tính mạng người bệnh. Khi chuyên gia nước ngoài hội chẩn với bác sĩ/bệnh nhân Việt Nam, rào cản ngôn ngữ gây ra sai sót nghiêm trọng về liều lượng thuốc, chỉ định phẫu thuật và làm chậm trễ quy trình cấp cứu. Các giải pháp truyền thống như Google Translate hay máy dịch tiêu dùng cầm tay thất bại do không có mạng trong phòng mổ/tầng hầm, độ trễ kéo dài nhiều giây, vi phạm bảo mật dữ liệu bệnh án và dịch sai lệch khi bác sĩ nói chèn tiếng Anh (như *"shock phản vệ"*, *"test PCR"*, *"đặt nội khí quản"*).

### 2.2 Phân tích tác động (Impact Analysis)

| Lĩnh vực Tác động | Nỗi đau Hiện tại (Current Pain Point) | Hệ quả Lâm sàng & Vận hành |
| :--- | :--- | :--- |
| **An toàn & Vận hành Lâm sàng** | Nghe/hiểu sai liều lượng thuốc, tiền sử dị ứng và y lệnh phẫu thuật khẩn cấp. | Tai biến y khoa, sốc phản vệ, chẩn đoán sai lệch và đe dọa tính mạng bệnh nhân. |
| **Năng suất & Quy trình** | Độ trễ phản hồi của app đám mây (>4–6s) làm đứt gãy luồng xử lý cấp cứu. | Ùn tắc quy trình tiếp nhận bệnh nhân, gián đoạn chuyển giao ca trực. |
| **Kết nối & Tính sẵn sàng** | Tầng hầm bệnh viện, phòng mổ cách ly và vùng sâu vùng xa không có internet ổn định. | Ứng dụng dịch thuật thông thường bị tê liệt hoàn toàn khi xảy ra sự cố khẩn cấp. |
| **Bảo mật Dữ liệu & Pháp lý** | Âm thanh giọng nói và bệnh án bị gửi lên máy chủ Cloud bên thứ ba. | Vi phạm quy định bảo mật thông tin y tế bệnh nhân (HIPAA/Nghị định 13), rủi ro pháp lý. |

### 2.3 Phân khúc Người dùng & Bối cảnh Sử dụng (Target Users & Use Cases)

| Phân khúc Người dùng | Nhu cầu Ngôn ngữ | Bối cảnh & Quy trình Trọng tâm | Mức độ Ưu tiên |
| :--- | :--- | :--- | :--- |
| **Bác sĩ & Điều dưỡng Việt Nam** | VI ↔ EN (Code-Switching) | Phân loại cấp cứu, bàn giao ca trực, theo dõi chỉ số sinh tồn ICU. | **Cực kỳ Cấp thiết** |
| **Bác sĩ Chuyên gia Nước ngoài** | EN ↔ VI | Khám hội chẩn chuyên sâu, mổ thị phạm, chuyển giao kỹ thuật. | **Cực kỳ Cấp thiết** |
| **Bệnh nhân & Du khách Quốc tế** | EN ↔ VI | Làm thủ tục nhập viện, khai báo triệu chứng, nhận hướng dẫn xuất viện. | **Cao** |
| **Nhân viên Cứu hộ / Paramedic** | VI ↔ EN | Vận chuyển xe cấp cứu, trạm y tế dã chiến, ứng phó thiên tai. | **Cao** |

### 2.4 Ràng buộc Thiết kế Cốt lõi (Key Design Constraints)

| Ràng buộc Kỹ thuật | Chỉ tiêu Kỹ thuật Đích / Yêu cầu Kỹ thuật |
| :--- | :--- |
| **Phụ thuộc Internet** | **Bằng Không (0%)** — Hoàn toàn tự chủ On-Device tại thời gian thực (Offline Runtime). |
| **Độ trễ Dịch thuật Đầu-cuối** | **< 1.4 giây** (Tổng thời gian từ khi ngắt giọng nói đến khi phát âm thanh dịch). |
| **Môi trường Triển khai** | Phòng cấp cứu, phòng mổ, xe cứu thương (Mức độ ồn âm học từ 65–80 dB). |
| **Cặp Ngôn ngữ & Phương ngữ** | Tiếng Việt ↔ Tiếng Anh (Xử lý giọng Bắc/Trung/Nam và chèn thuật ngữ tiếng Anh). |
| **Phần cứng & Năng lượng** | Thiết bị đeo ngực/lanyard rảnh tay, vỏ kháng khuẩn IP54, thời lượng pin **>10 giờ** liên tục. |

---

## 3. Giải Pháp Kinh Doanh & Đổi Mới Sáng Tạo
*(Trọng số Đánh giá: 15% — Business Solutions)*

### 3.1 Vấn đề Ngành & Sự Phù hợp của Giải pháp (Problem & Solution Fit)
Các ứng dụng dịch thuật di động (Google Translate, Microsoft Translator) và thiết bị dịch tiêu dùng (Pocketalk, Vasco) hoàn toàn không đáp ứng được tiêu chuẩn lâm sàng vì 3 lý do cốt tử:
1. **Bắt buộc có kết nối Internet**, biến thành "cục gạch" trong phòng mổ, tầng hầm hoặc vùng thiên tai cứu trợ.
2. **Độ trễ quá lớn (4–7 giây)**, làm gián đoạn nhịp trao đổi trong tình huống sinh tử và gây ức chế cho y bác sĩ.
3. **Sử dụng mô hình tổng quát**, thường xuyên dịch sai tên thuốc, chỉ số sinh tồn và thuật ngữ y tế chuyên ngành.

**MediVoice Edge** giải quyết triệt để khoảng trống này bằng một thiết bị Edge AI chuyên dụng. Bằng cách triển khai các mô hình ngôn ngữ thu nhỏ (SLMs) đã được chưng cất tri thức y khoa và tối ưu hóa chạy trực tiếp trên NPU Qualcomm Hexagon, MediVoice Edge đảm bảo hoạt động độc lập không cần mạng, phản hồi tức thì dưới 1.4s và chuẩn hóa hoàn toàn danh mục từ vựng y tế ICD-10.

### 3.2 Tính Đổi mới & Năng lực Cạnh tranh (Innovation & Competitive Strengths)

| Tiêu chí So sánh | Giải pháp Hiện có (Cloud / Thiết bị Tiêu dùng) | MediVoice Edge (Giải pháp của Chúng tôi) |
| :--- | :--- | :--- |
| **Tính Kết nối** | Bắt buộc 4G/5G/Wi-Fi; tê liệt trong phòng kín/tầng hầm cách ly sóng. | **100% Offline Trên Thiết bị** — Không cần Internet, bảo mật tuyệt đối. |
| **Độ trễ Phản hồi** | > 4–6 giây do truyền tải qua mạng và xếp hàng xử lý Cloud. | **< 1.4 giây đầu-cuối** nhờ pipeline streaming tăng tốc trên NPU tại chỗ. |
| **Độ chính xác Ngành Y** | Mô hình tổng quát; thường xuyên dịch sai tên bệnh, tên thuốc, y lệnh. | **Fine-tune chuyên sâu trên MedEV, VietMed & ViMedCSS;** từ điển G2P thuốc. |
| **Bảo mật Bệnh án** | Âm thanh và văn bản bị truyền ra ngoài máy chủ bên thứ ba. | **100% Xử lý Local;** không có dữ liệu nào rời khỏi thiết bị (chuẩn HIPAA). |
| **Chống ồn Môi trường** | Micro đơn giản; nhận diện kém trong tiếng còi, máy thở ồn ào. | **Mảng 3 Micro Beamforming + thuật toán RNNoise** triệt ồn 75+ dB. |

> **Tuyên bố Đột phá:** MediVoice Edge là thiết bị dịch thuật y tế bằng giọng nói biệt lập (air-gapped) đầu tiên được thiết kế riêng cho quy trình cấp cứu lâm sàng, kết hợp mô hình ngôn ngữ nhỏ SLM chưng cất tri thức với chip xử lý thần kinh NPU Hexagon của Qualcomm nhằm đạt độ trễ dưới 1.4s ở trạng thái không có Internet.

---

## 4. Tiếp Cận AI & Thiết Kế Kỹ Thuật
*(Trọng số Đánh giá: 35% — Tiêu chí Chiếm Tỉ trọng Cao nhất)*

### 4.1 Tổng quan Pipeline Hệ thống (System Pipeline Overview)
MediVoice Edge áp dụng kiến trúc pipeline nối tiếp dạng phân luồng (Cascaded Streaming Pipeline) gồm 3 module chuyên biệt thay vì dùng mô hình End-to-End nguyên khối. Cấu trúc này cho phép kiểm soát chặt chẽ dung lượng RAM, dễ dàng tinh chỉnh từ vựng y tế và khai thác tối đa tập lệnh gia tốc phần cứng của Qualcomm NPU.

```
+---------------------------------------------------------------------------------------------------+
|                              LUỒNG XỬ LÝ THỜI GIAN THỰC MEDIVOICE EDGE                            |
+---------------------------------------------------------------------------------------------------+
|  [ Mảng 3-Micro MEMS ]                                                                            |
|         │ (Tín hiệu âm thanh PCM 16 kHz / 16-bit)                                                 |
|         ▼                                                                                         |
|  [ Tiền xử lý Âm thanh & Triệt ồn ] ──► RNNoise / WebRTC NS (Định hướng chùm sóng Dual-mic)       |
|         │                                                                                         |
|         ▼                                                                                         |
|  [ Nhận diện Tiếng nói VAD ]        ──► Silero-VAD v4 (Phân đoạn khung âm thanh <10ms)            |
|         │                                                                                         |
|         ▼                                                                                         |
|  [ Động cơ Nhận diện ASR (Nghe) ]   ──► Distil-Whisper-Vi (VietMed/ViMedCSS) & Distil-Whisper-En  |
|         │                               (Lượng tử hóa INT8 trên Qualcomm NPU, Độ trễ <380ms)      |
|         ▼                                                                                         |
|  [ Dịch thuật Y khoa NMT / SLM ]    ──► Chưng cất Llama-3.2-1B-Medical / Qwen2.5-0.5B-Medical     |
|         │                               (INT4 AWQ trên NPU Hexagon, Speculative Decode <520ms)    |
|         ▼                                                                                         |
|  [ Tổng hợp Tiếng nói TTS (Nói) ]   ──► Piper-TTS / VITS-Edge (Bộ giọng Tiếng Việt & Tiếng Anh)   |
|         │                               (INT8 ONNX-QNN, Độ trễ <280ms, MOS > 4.25)                |
|         ▼                                                                                         |
|  [ Đầu ra Kết quả ]                 ──► IC Khuếch đại Class-D & Loa / Tai nghe BLE + Màn hình OLED|
+---------------------------------------------------------------------------------------------------+
|  TỔNG ĐỘ TRỄ DỊCH THUẬT ĐẦU-CUỐI: ~1.35 Giây (RTF < 0.55 trên Chip Qualcomm QCS6490 NPU)          |
+---------------------------------------------------------------------------------------------------+
```

### 4.2 Thiết kế Chi tiết Từng Module (Module-by-Module Design)

| Module Pipeline | Mô hình / Framework | Kích thước (Ước tính) | Độ trễ Đích | Kỹ thuật Tối ưu Cốt lõi |
| :--- | :--- | :--- | :--- | :--- |
| **Khử ồn & VAD** | Silero-VAD v4 + RNNoise (WebRTC DSP) | ~1.5 MB | < 10 ms / frame | Định hướng chùm sóng (Beamforming), phân đoạn âm thanh streaming. |
| **Nhận diện ASR (Nghe)** | Distil-Whisper-Small.en + Distil-Whisper-Vi ¹ (Fine-tune VietMed) | ~340 MB (Gộp INT8) | < 380 ms (chunk 500ms) | Biên dịch Qualcomm QNN INT8, giải mã lai CTC/Attention, hỗ trợ Code-switching. |
| **Dịch thuật NMT / SLM** | Llama-3.2-1B-Medical ² / Qwen2.5-0.5B-Medical ² | ~680 MB (INT4 AWQ) | < 520 ms (first token <120ms) | Lượng tử hóa 4-bit AWQ, Speculative Decoding, nhúng từ điển ICD-10. |
| **Tổng hợp TTS (Nói)** | Piper-TTS / VITS-Edge (Tiếng Việt & Tiếng Anh) | ~65 MB (ONNX-QNN) | < 280 ms (RTF 0.18) | Vector hóa INT8 trên HTP, vocoder streaming tái tạo phổ Mel tự nhiên. |

> ¹ **Distil-Whisper-Vi:** Phiên bản chưng cất tri thức (Knowledge Distillation) từ PhoWhisper-large (VinAI, ICLR 2024) do đội tự thực hiện bằng kỹ thuật LoRA fine-tuning trên tập dữ liệu VietMed + ViMedCSS, tốc độ suy luận nhanh hơn ~5.6× so với Whisper-small gốc với WER chênh lệch < 1%.

> ² **"Llama-3.2-1B-Medical" và "Qwen2.5-0.5B-Medical"** là tên nội bộ (internal name) của đội cho các mô hình SLM được đội tự fine-tune từ `meta-llama/Llama-3.2-1B-Instruct` và `Qwen/Qwen2.5-0.5B-Instruct` (HuggingFace) bằng kỹ thuật QLoRA 4-bit + Knowledge Distillation từ teacher model lớn hơn, trên tập ngữ liệu y tế song ngữ MedEV (~360K cặp câu) và FutureBeeAI.

### 4.3 Tối ưu hóa On-Device & Quản lý Bộ nhớ RAM
1. **Lượng tử hóa & Biên dịch qua Qualcomm AI Hub:** Tất cả các mô hình được chuyển đổi sang ONNX, áp dụng kỹ thuật Lượng tử hóa sau huấn luyện (PTQ) và QAT xuống INT8 (ASR, TTS) và INT4 AWQ (SLM), sau đó biên dịch thành định dạng Deep Learning Container (`.dlc`) tối ưu cho Qualcomm Hexagon Tensor Processor (HTP).
2. **Kiến trúc Bộ nhớ Zero-Copy:** Tận dụng vùng đệm chia sẻ ION / DMA giữa CPU, DSP và NPU, giúp truyền tensor âm thanh và embedding văn bản trực tiếp trong RAM mà không cần copy dữ liệu qua lại giữa các tiến trình.
3. **Phân bổ Bộ nhớ RAM Chi tiết:** ASR thường trú (~340 MB) + SLM (~680 MB) + TTS (~65 MB) + Vùng đệm Audio & Hệ điều hành Linux (~1.05 GB) = **Tổng mức chiếm dụng RAM ~2.14 GB**, hoàn toàn nằm trong giới hạn an toàn của thanh RAM 8 GB LPDDR4x.

### 4.4 Độ bền vững, Xử lý Tình huống Ngoại lệ & Rủi ro Kỹ thuật

**a) Xử lý tình huống ngoại lệ:**
1. **Chống ồn Môi trường Y tế:** Huấn luyện mô hình với kỹ thuật Noise Augmentation (chèn tiếng ồn còi cấp cứu, máy thở, tiếng bước chân từ ESC-50 và AudioSet ở tỉ lệ SNR 5–15 dB) kết hợp thuật toán lọc nhiễu 3 micro.
2. **Xử lý Code-Switching & Tiếng lóng Y khoa:** Mở rộng Tokenizer với hơn 2,400 thuật ngữ y tế ICD-10 và ngữ liệu ViMedCSS. Sử dụng từ điển chuyển đổi âm vị G2P (Grapheme-to-Phoneme) để chuẩn hóa cách phát âm tên thuốc tiếng Anh theo kiểu người Việt (ví dụ: *"Paracetamol"*, *"Panadol"*, *"Aspirin"*).
3. **Cơ chế Dự phòng An toàn:** Tích hợp bộ nhớ đệm phản xạ nhanh (Instant Flash-Cache) lưu sẵn hơn 100 câu lệnh cấp cứu tiêu chuẩn (như *"Kiểm tra mạch"*, *"Bệnh nhân sốc phản vệ"*) để phát âm thanh dịch ngay dưới 50ms khi gặp tình huống nguy cấp.

**b) Rủi ro Kỹ thuật & Phương án Giảm thiểu (Technical Risks & Mitigations):**

| Rủi ro Kỹ thuật | Khả năng Xảy ra | Phương án Giảm thiểu (Mitigation) |
| :--- | :---: | :--- |
| **Vượt giới hạn VTCM memory NPU** trên QCS6490 khi tải đồng thời ASR + SLM | Trung bình | Chuyển sang mô hình dự phòng **Qwen2.5-0.5B** (gọn hơn ~40%), giảm kích thước KV-Cache, phân tách tải mô hình theo pipeline. |
| **WER > 20%** với giọng miền Trung và phương ngữ lẫn Code-Switching phức tạp | Trung bình | Tăng tỉ lệ dữ liệu giọng Trung trong tập huấn luyện tối thiểu 25%; áp dụng Accent Adversarial Training và CTC beam search với vocabulary bias. |
| **Latency spike > 1.4s** khi xử lý câu dài (>15 từ) trong môi trường nhiễu cao | Thấp | Kích hoạt chế độ **Chunk-Priority Mode** (chia nhỏ câu thành 2 đoạn ≤7 từ); fallback phát ngay flash-cache nếu độ tin cậy (confidence score) <0.65. |
| **Tiêu thụ pin vượt ngưỡng** khi NPU + loa chạy đồng thời liên tục >8 giờ | Thấp | Chế độ **Adaptive Power Throttle**: giảm tần số NPU xuống 80% sau 6 giờ hoạt động, tắt màn hình OLED ở chế độ lắng nghe liên tục. |

### 4.5 Benchmark Cơ Sở & Kế Hoạch Đánh Giá
*(Phần này đáp ứng yêu cầu "accuracy targets and evaluation plan" của Tiêu chí 1 — Trọng số 35%)*

#### 4.5.1 Benchmark Cơ Sở từ Nghiên cứu (Baseline từ Literature)

**Module ASR — Nhận diện Tiếng nói Tiếng Việt:**

| Mô hình | Tập Kiểm tra | WER (%) | Nguồn |
| :--- | :--- | :---: | :--- |
| XLSR-53 (không fine-tune) | VietMed test | 51.8% | LREC-COLING 2024 |
| XLSR-53-Viet (fine-tune y tế) | VietMed test | 29.6% | LREC-COLING 2024 |
| PhoWhisper-small (VinAI) | CMV-Vi | 11.08% | ICLR 2024 Tiny Papers |
| PhoWhisper-small (VinAI) | VIVOS | 6.33% | ICLR 2024 Tiny Papers |
| Distil-Whisper-small.en | LibriSpeech test-clean | 10.0% | HuggingFace 2023 |
| **MediVoice Edge ASR (Mục tiêu)** | **VietMed medical test** | **< 15%** | **Đội đặt mục tiêu** |

**Module NMT/SLM — Dịch thuật Y khoa VI ↔ EN:**

| Mô hình | Chiều Dịch | BLEU | COMET | Nguồn |
| :--- | :--- | :---: | :---: | :--- |
| OPUS-MT Helsinki (general) | EN → VI | ~37.2 | — | Tatoeba benchmark |
| vinai-translate (fine-tune MedEV) | EN ↔ VI | SOTA | — | LREC-COLING 2024 |
| SLM 1B–3B fine-tune MedEV (tham khảo) | VI → EN | Cạnh tranh với model lớn | > 0.80 | VLSP 2025 Shared Task |
| **MediVoice Edge SLM (Mục tiêu)** | **VI ↔ EN (medical)** | **> 42** | **> 0.82** | **Đội đặt mục tiêu** |

**Module TTS — Chất lượng Giọng Tổng hợp:**

| Hệ thống | MOS (Mean Opinion Score) | Nguồn |
| :--- | :---: | :--- |
| VITS streaming Vietnamese (nghiên cứu) | 4.35 – 4.43 | ISCA Archive / INTERSPEECH |
| Piper `vi_VN-vais1000-medium` (offline) | ~3.8 – 4.0 | Community evaluation |
| **MediVoice Edge TTS (Mục tiêu)** | **> 4.1** | **Human evaluation panel** |

#### 4.5.2 Phân bổ Ngân sách Độ trễ Đầu-cuối (Latency Budget)

| Giai đoạn Pipeline | Thời gian Đích | Cơ sở Ước tính |
| :--- | :---: | :--- |
| VAD + Noise Suppression | < 10 ms | Silero-VAD v4 benchmark |
| ASR (chunk 500ms, INT8 QNN) | < 380 ms | QCS6490 QNN encoder ~250ms (community), decoder tăng tốc |
| NMT/SLM (INT4 AWQ, speculative) | < 520 ms | First token < 120ms; speculative decode tăng 2.5× |
| TTS (INT8 ONNX-QNN, streaming) | < 280 ms | RTF 0.18 trên HTP |
| **Tổng cộng (Đầu-cuối)** | **~1.35 giây** | **RTF < 0.55 trên QCS6490 NPU** |

#### 4.5.3 Kế hoạch Đánh giá Phase 3 (Evaluation Plan)
Sau khi triển khai prototype trên thiết bị QCS6490, đội sẽ thực hiện đánh giá theo 3 tầng:

1. **Đánh giá Tự động (Automated):** Đo WER trên VietMed test set, BLEU/COMET trên MedEV test set (5,000 cặp câu), RTF và peak RAM trên thiết bị.
2. **Đánh giá Giọng nói Tổng hợp (MOS):** Khảo sát MOS với nhóm 10 người đánh giá (5 bác sĩ + 5 người dùng thông thường) trên 50 câu lâm sàng mẫu.
3. **Kiểm thử Môi trường Thực địa (Field Simulation):** Mô phỏng phòng cấp cứu với tiếng ồn nền 70 dB (ESC-50), đo WER và độ trễ thực tế trong điều kiện nhiễu.

---

## 5. Phần Cứng & Ý Niệm Thiết Bị
*(Trọng số Đánh giá: 25% — Hardware & Device Concept)*

### 5.1 Lựa chọn Nền tảng SoC & Biện giải (Platform Selection)

| Tiêu chí Đánh giá | Qualcomm QCS6490 (RB3 Gen 2) | NVIDIA Jetson Orin Nano | Raspberry Pi 5 + Hailo-8 |
| :--- | :--- | :--- | :--- |
| **Hiệu năng NPU (TOPS)** | **12–13 TOPS (Hexagon HTP)** | 20–40 TOPS (Ampere GPU) | 13–26 TOPS (Hailo NPU) |
| **Công suất Tiêu thụ (TDP)** | **5.0 – 7.0 W (Được chọn: 5.2 W peak)** | 10.0 – 15.0 W (Tỏa nhiệt cao) | 8.0 – 12.0 W (Trung bình) |
| **Kích thước & Trọng lượng** | **Siêu nhỏ gọn / Đeo ngực (<150g)** | Cồng kềnh / Tản nhiệt nặng (>350g) | Dạng module ghép nối (>220g) |
| **Hệ sinh thái & Công cụ SDK** | **Qualcomm QNN, SNPE, AI Hub (Được chọn)** | NVIDIA TensorRT | Hailo TAPPAS / PyTorch ONNX |
| **Trạng thái Lựa chọn** | **ĐƯỢC CHỌN [✓]** | Loại bỏ [—] | Loại bỏ [—] |

**Biện giải Lựa chọn Nền tảng:** Qualcomm Snapdragon QCS6490 (RB3 Gen 2) được chọn vì mang lại hiệu suất năng lượng NPU cao nhất (13 TOPS với công suất đỉnh <5.5W), hỗ trợ tăng tốc phần cứng nguyên bản cho INT4/INT8 trên Hexagon HTP, tích hợp sẵn DSP xử lý âm thanh và kích thước siêu nhẹ lý tưởng cho thiết bị đeo ngực y tế chuyên dụng (<150g, chuẩn kháng khuẩn IP54).

### 5.2 Linh kiện Phần cứng Cốt lõi & Phân bổ Công suất (Hardware BOM & Power Budget)

| Linh kiện Phần cứng | Thông số Kỹ thuật Chi tiết | Công suất Đỉnh | Ghi chú Vận hành |
| :--- | :--- | :--- | :--- |
| **Module Vi xử lý SoC** | Qualcomm QCS6490 (Kryo CPU, Adreno GPU, Hexagon NPU) | 3.8 W | Xử lý đồng thời NPU + CPU khi dịch thuật |
| **Mảng Micro (Mic Array)** | 3× Micro MEMS đa hướng (16 kHz, 64 dB SNR) | 0.08 W | Lọc nhiễu phần cứng & định hướng chùm sóng |
| **Âm thanh / Loa Phát** | IC khuếch đại Class-D + Loa 1.5W / Bluetooth 5.3 | 0.65 W | Phát âm thanh rõ ràng; hỗ trợ tai nghe BLE |
| **Màn hình Hiển thị** | Màn hình 2.13-inch E-Paper / OLED tiết kiệm điện | 0.15 W | Hiển thị văn bản dịch đối chiếu trực quan |
| **Bộ nhớ RAM & Bộ nhớ trong** | 8 GB LPDDR4x + 64 GB UFS 3.1 | 0.35 W | Chứa toàn bộ hệ điều hành & file model .dlc |
| **Hệ thống Pin** | Pin Li-Po mật độ cao 3,500 mAh (3.8V) | — | **Đảm bảo > 10.5 giờ** ca trực liên tục |
| **Tổng Công suất Đỉnh** | Toàn bộ module chạy hết công suất dịch & phát | **5.03 W** | Công suất chờ/nghe trung bình: ~0.85 W |

---

## 6. Kiến Trúc Hệ Thống & Tích Hợp

### 6.1 Ngăn xếp Phần mềm (Software Stack)

| Tầng Phần mềm | Công nghệ / Framework Sử dụng | Vai trò Chức năng trong MediVoice Edge |
| :--- | :--- | :--- |
| **Tầng Ứng dụng & UX** | C++20 Core Orchestrator + Embedded Qt/QML UI | Điều phối luồng sự kiện, nút bấm PTT, hiển thị màn hình OLED. |
| **Tầng Thực thi AI** | Qualcomm Neural Processing SDK (QNN) & ONNX EP | Nạp và thực thi các model ASR, MT, TTS trên NPU Hexagon. |
| **Tầng Xử lý Âm thanh** | TinyALSA + WebRTC AudioProcessing / RNNoise | Khử tiếng vọng (AEC), lọc tạp âm và phân đoạn giọng nói VAD. |
| **Tầng Bộ nhớ Đệm** | Zero-Copy Circular Ring-Buffer & Flash Cache | Truyền token dạng streaming và phát nhanh cụm từ khẩn cấp (<50ms). |
| **Hệ điều hành Cơ sở** | Custom Yocto Linux (Kernel 6.1 LTS vá Real-Time) | Lập lịch xử lý âm thanh thời gian thực và quản lý nguồn điện tối ưu. |

### 6.2 Nguyên tắc Thiết kế Offline-First (Độc lập Hoàn toàn)
1. **Không Phụ thuộc Mạng:** 100% trọng số mô hình, bộ từ điển âm vị và engine giải mã âm thanh được lưu trực tiếp trên bộ nhớ flash 64 GB UFS 3.1. Thiết bị hoàn toàn không có luồng truyền dữ liệu ra ngoài, đảm bảo tính biệt lập tuyệt đối (Air-gapped).
2. **Biên dịch Trước Tối ưu NPU:** Toàn bộ model được biên dịch sẵn sang file nhị phân `.dlc` của Qualcomm từ giai đoạn build, triệt tiêu thời gian khởi tạo lúc runtime và đảm bảo phân bổ RAM cố định, không rò rỉ bộ nhớ.
3. **Xử lý Streaming Độ trễ Thấp:** Các khung âm thanh (16 kHz, 16-bit mono) được xử lý theo cửa sổ trượt 500ms, cho phép động cơ dịch thuật bắt đầu dịch ngay khi người dùng vẫn đang nói câu dài.

---

## 7. Hồ Sơ Đội Ngũ & Kế Hoạch Triển Khai
*(Trọng số Đánh giá: 10% — Team and Execution Plan)*

### 7.1 Thành viên Đội ngũ & Phân công Trách nhiệm

| Họ và Tên | Vai trò Đảm nhiệm | Chuyên môn Cốt lõi | Phạm vi Đóng góp trong Dự án |
| :--- | :--- | :--- | :--- |
| **Trịnh Quốc Bảo** | Trưởng nhóm & Kỹ sư Trưởng AI | Edge AI, NLP, Xử lý Tiếng nói | Thiết kế tổng thể pipeline, chưng cất mô hình ASR & MT, tối ưu NPU. |
| **Nguyễn Văn Thái** | Kỹ sư Machine Learning | Ngữ liệu Song ngữ, MT Benchmark | Thu thập & chuẩn hóa MedEV, FutureBeeAI, đánh giá BLEU/COMET. |
| **Phạm Đăng Khoa** | Kỹ sư Dữ liệu & Đảm bảo Chất lượng | Phân tích Dữ liệu Y tế, QA Systems | Phân tích ViMedCSS, QA y tế, khử trùng lặp dữ liệu & kiểm thử. |
| **Võ Hoàng Minh** | Kỹ sư Phần cứng Nhúng | Embedded Linux, Qualcomm SDK | Tích hợp phần cứng QCS6490, đo đạc công suất và lượng tử hóa QNN. |

### 7.2 Lộ trình Dự án & Các Cột mốc Quan trọng

| Giai đoạn | Cột mốc Dự án | Hoạt động Trọng tâm & Sản phẩm Bàn giao | Thời gian Đích |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Nghiên cứu & Khám phá Dữ liệu | Khảo sát bài toán y tế, EDA VietMed/MedEV, benchmark model Qualcomm AI Hub. | Tháng 6/2026 *(Hoàn thành)* |
| **Phase 2** | Đề xuất Kỹ thuật & Đặc tả Pipeline | Hoàn thiện thiết kế hệ thống, chiến lược lượng tử hóa, cấu trúc BOM & công suất. | Tháng 8/2026 *(Nộp hiện tại)* |
| **Phase 3** | Tích hợp Prototype Trên Thiết bị | Triển khai model DLC lên NPU QCS6490, viết pipeline C++, đo đạc WER/BLEU. | Tháng 9/2026 |
| **Phase 4** | Thử nghiệm Thực địa & Chung kết | Kiểm thử giả lập phòng cấp cứu, tối ưu hóa độ bền âm học, thuyết trình Chung kết. | Tháng 10–11/2026 |

---

## 8. Checklist Kiểm Tra Hồ Sơ Nộp Bài

| # | Hạng mục Kiểm tra | Trạng thái |
| :---: | :--- | :---: |
| 1 | Tóm tắt điều hành (Executive Summary) hoàn chỉnh (nêu rõ vấn đề, giải pháp, giá trị) | **[X] Đã Hoàn Thành** |
| 2 | Tuyên bố bài toán, Phân tích tác động, Đối tượng người dùng và Ràng buộc thiết kế đầy đủ | **[X] Đã Hoàn Thành** |
| 3 | Giải pháp kinh doanh hoàn thiện — nêu rõ khoảng trống ngành, tính phù hợp và khác biệt | **[X] Đã Hoàn Thành** |
| 4 | Pipeline AI được mô tả chi tiết kèm thông số model, độ trễ và chiến lược tối ưu NPU | **[X] Đã Hoàn Thành** |
| 5 | Lựa chọn phần cứng được biện giải rõ ràng; có bảng BOM và phân bổ công suất chi tiết | **[X] Đã Hoàn Thành** |
| 6 | Hồ sơ đội ngũ và lộ trình triển khai khả thi, bám sát các mốc thời gian cuộc thi | **[X] Đã Hoàn Thành** |
| 7 | Đã thay thế toàn bộ placeholder; sơ đồ kiến trúc, luồng dữ liệu và bộ nhớ được mô tả rõ | **[X] Đã Hoàn Thành** |
| 8 | Hồ sơ đã được xuất định dạng Word/Markdown sẵn sàng để export PDF nộp cổng dự thi | **[X] Đã Hoàn Thành** |
