# Đánh giá & Tổng hợp Dataset cho Dự án Dịch thuật Y tế (Vie ↔ Eng)

Để xây dựng một thiết bị dịch thuật Edge AI hoàn chỉnh, bạn không chỉ cần 2 bộ dữ liệu (Tiếng Anh và Tiếng Việt), mà thực tế hệ thống của bạn sẽ được chia thành 3 module chính. Mỗi module cần các bộ dataset đặc thù tập trung vào lĩnh vực Y tế (Healthcare):

1. **ASR (Automatic Speech Recognition - Nhận diện giọng nói):** Cần dữ liệu Audio Y tế Tiếng Việt và Tiếng Anh.
2. **MT (Machine Translation - Dịch máy):** Cần kho ngữ liệu song ngữ (Parallel Corpus) Tiếng Việt ↔ Tiếng Anh chuyên ngành Y.
3. **TTS (Text-To-Speech - Tổng hợp giọng nói):** Thường sử dụng các mô hình có sẵn hoặc fine-tune trên giọng đọc tiêu chuẩn.

Dưới đây là đánh giá sâu các bộ dataset tốt nhất hiện có cho dự án của bạn:

---

## 1. Dữ liệu Dịch máy (Machine Translation) - Ngữ liệu song ngữ Vie ↔ Eng

Module này quyết định "bộ não" chuyển đổi ngôn ngữ của thiết bị. Yêu cầu lớn nhất là tính chính xác của thuật ngữ y khoa.

### 🌟 MedEV (Đã được chọn bởi Thái)
* **Lưu ý:** Dataset này đã được teammate (Thái) chọn, nên chúng ta cần chuyển sang các phương án thay thế bên dưới.
* **Quy mô:** ~360,000 cặp câu song ngữ (sentence pairs).
* **Nguồn gốc:** Nghiên cứu học thuật, được trích xuất và tinh chỉnh đặc biệt cho Medical Domain.
* **Đánh giá:** 
  * *Ưu điểm:* Là bộ dữ liệu Medical Parallel Corpus mở (public) lớn nhất và chất lượng nhất hiện nay cho cặp Anh-Việt.
  * *Nhược điểm:* Câu văn có xu hướng nặng tính hàn lâm (từ các tài liệu y khoa) hơn là các đoạn hội thoại giao tiếp thường ngày.

### 🚀 FutureBeeAI English-Vietnamese Medical Parallel Corpus (Đề xuất thay thế cho MT)
* **Quy mô:** Hơn 50,000 cặp câu song ngữ.
* **Đặc điểm:** Thiết kế chuyên biệt cho các ứng dụng Medical NLP, bao gồm cả dịch máy và nhận diện ý định.
* **Đánh giá:** Đây là giải pháp thay thế hoàn hảo nhất cho MedEV để huấn luyện song ngữ Anh-Việt chuyên ngành Y mà chưa bị trùng lặp với các team khác.

### 💬 hungsvdut2k2/vietnamese-medical-chat-data (Hugging Face)
* **Quy mô:** ~46,500 bản ghi hội thoại y tế.
* **Đánh giá:** Rất hữu ích nếu thiết bị Edge AI của bạn hướng đến giao tiếp tự nhiên và cần fine-tune model cho bối cảnh chat/hỏi đáp thay vì chỉ dịch thuật văn bản hàn lâm. Có thể phối hợp với bộ HoangHa ở dưới.

### 🏥 QA Datasets: hungnm/vietnamese-medical-qa & tmnam20/ViMedAQA
* **Quy mô:** Hơn 9,000 cặp QA (hỏi đáp) từ Vinmec, eDoctor (bộ hungnm) và các bài viết y khoa (bộ ViMedAQA).
* **Đánh giá:** Giải pháp tuyệt vời để bù đắp việc Khoa đã lấy bộ **MedQuad + Meddies Consultant**. Bạn có thể dùng bộ này để fine-tune cho tác vụ y tế tại Việt Nam nếu thiết bị cần chức năng tư vấn lâm sàng.

### 📊 HoangHa/medical-data (Hugging Face)
* **Đặc điểm:** Dữ liệu hội thoại lâm sàng (clinical dialogue), hỗ trợ song ngữ.
* **Đánh giá:** 
  * Cực kỳ phù hợp cho bối cảnh "hội thoại" thực tế, giúp hệ thống dịch tự nhiên hơn (đạt điểm MOS cao hơn trong phần Naturalness).

### 🧬 ViPubmed / ViPubmedT5 Data
* **Quy mô:** Hàng triệu bản ghi tổng hợp (Synthetic Data).
* **Đánh giá:** Dịch tự động từ các bản tóm tắt PubMed tiếng Anh sang tiếng Việt. Phù hợp để pre-train model cho quen với các thuật ngữ y khoa khó (như tên thuốc, tên bệnh hiếm), nhưng có thể bị nhiễu do là dữ liệu dịch máy.

---

## 2. Dữ liệu Nhận diện Giọng nói (ASR) - Tiếng Việt Y tế

Module này thu âm bác sĩ/bệnh nhân người Việt và chuyển thành text. Đây là bài toán khó do tiếng Việt có dấu và nhiễu môi trường.

### 🌟 VietMed (Cực kỳ khuyên dùng)
* **Quy mô:** 16 giờ âm thanh y khoa đã gán nhãn (labeled) + 1,000 giờ âm thanh y khoa chưa gán nhãn + 1,200 giờ âm thanh tổng quát.
* **Nguồn gốc:** Bài báo tại LREC (chuyên trang ngôn ngữ học).
* **Đánh giá:**
  * *Ưu điểm:* Được xem là bộ dataset ASR Y tế Tiếng Việt chuẩn mực nhất hiện nay. Bao phủ toàn bộ nhóm bệnh ICD-10, có sự đa dạng về giọng vùng miền (Bắc/Trung/Nam) và điều kiện ghi âm. Đã có sẵn model pre-train như `w2v2-Viet`.
  * *Nhược điểm:* Dữ liệu gán nhãn (16 giờ) có thể chưa đủ lớn nếu training từ scratch, nên tận dụng phương pháp self-supervised learning trên 1,000 giờ un-labeled.

### 🗣️ ViMedCSS (Code-Switching Y tế)
* **Đặc điểm:** Tập trung vào hiện tượng "Code-Switching" (Pha trộn tiếng Việt và tiếng Anh) - cực kỳ phổ biến trong bệnh viện ở VN khi bác sĩ chèn các thuật ngữ tiếng Anh vào câu tiếng Việt (ví dụ: "bệnh nhân bị *shock phản vệ*, cần *test* nhanh").
* **Đánh giá:** Giải quyết trực tiếp pain-point thực tế của dự án. Nên dùng để fine-tune để tránh model nhận diện sai các từ tiếng Anh bị chèn vào.

---

## 3. Dữ liệu Nhận diện Giọng nói (ASR) - Tiếng Anh Y tế

Dùng để nhận diện giọng nói của bác sĩ/chuyên gia nước ngoài.

### 🎧 Eka Medical ASR Evaluation Dataset (Hugging Face)
* **Quy mô:** > 3,900 bản ghi âm tiếng Anh.
* **Đánh giá:** Tập trung vào thuật ngữ y khoa, câu kể và giao tiếp hội thoại. Là nguồn public tốt và dễ tiếp cận nhất.

### 🩺 Medical Speech, Transcription, and Intent (Kaggle)
* **Quy mô:** ~8.5 giờ hội thoại y tế.
* **Đánh giá:** Gồm các câu nói về triệu chứng chung. Dễ dùng để build prototype ban đầu, nhưng có thể cần lọc lại vì chất lượng (noise) không đồng đều.

### 🎙️ MedDialogue-Audio (Synthetic Data)
* **Đánh giá:** Vì lý do bảo mật quyền riêng tư (HIPAA), dữ liệu ghi âm y tế thật của Mỹ rất hiếm. Bộ này dùng mô hình tổng hợp giọng nói để đọc các kịch bản hội thoại y tế và chèn thêm tiếng ồn nền (background noise) của bệnh viện. Rất tốt để rèn luyện độ "lì" (robustness) cho model trong môi trường ồn.

---

## 💡 Lời khuyên chiến lược cho Dự án (Project Strategy)

1. **Khởi đầu (Prototype):** Đừng train từ đầu. Hãy tải các model từ **Qualcomm AI Hub** (đã được nén và tối ưu cho Edge) như Whisper (cho ASR) hoặc Llama cỡ nhỏ (cho Translation). 
2. **Fine-tuning:**
   * Lấy **FutureBeeAI Medical Parallel Corpus** và **hungsvdut2k2/vietnamese-medical-chat-data** để fine-tune khả năng dịch thuật (MT) nhằm tối ưu cho văn phong hội thoại y tế thay vì chỉ dịch thuật văn bản (do MedEV đã bị chọn).
   * Lấy **VietMed** và **ViMedCSS** để fine-tune khả năng nghe (ASR) cho tiếng Việt, đặc biệt tập trung vào việc nhận diện "Code-switching" (đọc lẫn lộn Việt-Anh).
   * Lấy **Eka Medical ASR** hoặc **Medical Speech, Transcription, and Intent** để fine-tune khả năng nghe (ASR) cho tiếng Anh, giúp mô hình bắt chính xác các thuật ngữ y khoa do chuyên gia/bác sĩ nước ngoài phát âm.
3. **Mô phỏng môi trường:** Tại vòng chung kết cuộc thi (Tháng 11), ban giám khảo có thể test trong môi trường ồn ào. Hãy augment dữ liệu bằng cách tự mix các audio ASR với tiếng ồn bệnh viện (còi xe cấp cứu, tiếng máy thở, tiếng bước chân) để model của bạn không bị "điếc" thực tế.
