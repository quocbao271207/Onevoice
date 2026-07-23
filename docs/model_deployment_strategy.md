# Chiến lược Triển khai Mô hình Edge AI (Healthcare Translation)

Để đáp ứng tiêu chí **hoàn toàn On-device (0% API Calls)**, độ trễ cực thấp (RTF < 1.0, Turnaround Latency < 2.0s) và chạy trên phần cứng giới hạn (điện thoại hoặc bo mạch), hệ thống không thể sử dụng các mô hình nguyên bản. Dưới đây là chiến lược cấu trúc pipeline, lựa chọn mô hình và các kỹ thuật tối ưu hóa (đặc biệt là Distillation).

---

## 1. Cấu trúc Pipeline

Có 2 hướng tiếp cận chính:
1. **End-to-End (Speech-to-Speech):** Dịch thẳng từ âm thanh sang âm thanh (như SeamlessM4T). *Nhược điểm:* Quá nặng cho thiết bị Edge, khó tinh chỉnh từ vựng y tế, độ trễ cao.
2. **Cascaded (Khuyên dùng):** Nối tiếp 3 module: `ASR -> MT -> TTS`. *Ưu điểm:* Dễ kiểm soát độ trễ, dễ thay thế/fine-tune từng module chuyên biệt cho ngành Y tế. Có thể dùng kỹ thuật streaming (ASR nhận diện được chữ nào là MT dịch ngay chữ đó) để giảm Turnaround Latency.

---

## 2. Các hướng Mô hình Khả thi (Candidate Models)

Thay vì gọi API, chúng ta sẽ nhúng các model nhỏ (Small/Edge Models) vào thẳng thiết bị:

### A. Nhận diện giọng nói (ASR)
*   **Tiếng Anh:** `distil-whisper-small.en` hoặc `distil-whisper-base.en`. Đây là các phiên bản đã được *Distilled* (chưng cất) từ Whisper Large, chạy siêu nhanh (nhanh hơn 6 lần) nhưng giữ nguyên độ chính xác.
*   **Tiếng Việt:** 
    *   *Hướng 1:* Dùng `Wav2Vec2-XLSR-53` (có bản fine-tune tiếng Việt) - nhẹ và chuyên tiếng Việt.
    *   *Hướng 2:* Tự làm "Distil-Whisper" cho Tiếng Việt. Lấy bản Whisper-Tiny, fine-tune lại bằng tập **VietMed** và **ViMedCSS** (Code-switching).

### B. Dịch máy (Machine Translation - MT)
*   **Dùng SLMs (Small Language Models):** Các model dưới 3 Tỷ tham số (3B parameters) là lý tưởng.
    *   `Qwen2.5-0.5B` hoặc `Qwen2.5-1.5B`: Kích thước cực nhỏ (vài trăm MB khi lượng tử hóa), tư duy tốt.
    *   `Llama-3.2-1B`: Model siêu nhẹ mới nhất của Meta thiết kế riêng cho Edge/Mobile.
    *   *Chiến lược:* Fine-tune Llama-3.2-1B hoặc Qwen2.5-0.5B trên tập **MedEV** để ép nó chỉ chuyên dịch câu Y khoa Anh-Việt.
*   **Dùng NLLB (No Language Left Behind):** Bản `NLLB-200-600M` của Meta. Rất nhẹ và hỗ trợ tiếng Việt cực tốt.

### C. Tổng hợp giọng nói (TTS)
*   **VITS (Conditional Variational Autoencoder with Adversarial Learning for E2E TTS):** Cực kỳ nhanh, là tiêu chuẩn cho real-time TTS hiện nay.
*   **Piper TTS:** Một hệ thống TTS tối ưu hóa siêu độ trễ, thiết kế riêng để chạy cục bộ trên Raspberry Pi và điện thoại, không cần GPU mạnh.

---

## 3. Chiến thuật Tối ưu hóa (Distillation & Quantization)

Để nhét 3 module này vào một thiết bị di động và chạy mượt, ta BẮT BUỘC phải dùng các kỹ thuật sau:

### 🔬 3.1. Knowledge Distillation (Chưng cất tri thức)
Đúng như bạn nói, Distillation là chìa khóa. Ý tưởng là dùng một "Thầy" (Teacher - Model lớn như Llama-3 70B hoặc GPT-4) dạy cho "Trò" (Student - Model nhỏ như Llama-3.2-1B hoặc NLLB-600M).
*   **Cách làm:**
    *   Cho Teacher model dịch và tạo ra hàng chục nghìn đoạn hội thoại lâm sàng giả lập phức tạp (có chứa thuật ngữ y khoa khó).
    *   Yêu cầu Teacher xuất ra cả lý luận (chain-of-thought) về tại sao dịch từ đó.
    *   Lấy toàn bộ dữ liệu (Input - Output) của Teacher để train cho Student. 
    *   *Kết quả:* Student học được cách dịch thuật ngữ Y khoa chuẩn xác như GPT-4 nhưng kích thước chỉ bằng 1/100, đủ nhét vào RAM điện thoại.

### 🗜️ 3.2. Quantization (Lượng tử hóa)
*   Giảm trọng số của mô hình từ 32-bit (FP32) xuống 8-bit (INT8) hoặc 4-bit (INT4). Một model 1B tham số sẽ giảm dung lượng từ 4GB xuống còn khoảng 700MB.
*   **Công cụ chiến lược:** Vì giải do Qualcomm đồng tổ chức, điểm mấu chốt là bạn **PHẢI TÌM HIỂU Qualcomm AI Hub**. Họ cung cấp các bộ công cụ compile model để chạy khai thác tối đa sức mạnh của lõi NPU (Neural Processing Unit) trên chip Snapdragon thay vì chạy trên CPU/GPU thông thường. 
    *   Sử dụng: `Qualcomm Neural Processing SDK` (QNN) để convert model PyTorch/TensorFlow sang định dạng chuẩn của Snapdragon (.dlc).

### 🚀 3.3. Kỹ thuật giảm độ trễ (Latency Reduction)
*   **Streaming ASR:** Không đợi người dùng nói hết câu mới bắt đầu nhận diện. ASR nhận diện theo từng chunk âm thanh (vd 500ms).
*   **Streaming MT / Speculative Decoding:** Ngay khi ASR trả ra 3-4 từ tiếng Việt, module MT bắt đầu dịch nháp trước tiếng Anh. Kỹ thuật này giúp bạn dễ dàng đạt chuẩn *Turnaround Latency < 2.0s* của cuộc thi.
