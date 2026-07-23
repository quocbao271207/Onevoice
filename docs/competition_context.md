# Context Cuộc thi: OneVoice AI Challenge 2026

**Đơn vị tổ chức:** Saigon AI Hub × Qualcomm
**Mục tiêu cuộc thi:** Xây dựng giải pháp/thiết bị dịch thuật đa ngôn ngữ thời gian thực thế hệ mới, hoạt động **hoàn toàn trên thiết bị (On-device / Edge AI)** mà không cần kết nối Cloud/Internet.

---

## 🎯 Định hướng của Team

- **Cặp ngôn ngữ lựa chọn:** Tiếng Việt ↔ Tiếng Anh (Vietnamese ↔ English)
- **Lĩnh vực (Domain) ứng dụng:** Y tế / Chăm sóc sức khỏe (Healthcare)

**Bối cảnh bài toán Y tế (Healthcare Context):** 
Trong môi trường bệnh viện, phòng khám, các khu vực y tế, sự bất đồng ngôn ngữ giữa y bác sĩ (đặc biệt là chuyên gia nước ngoài) và bệnh nhân là một rào cản lớn. 
- Yêu cầu dịch thuật thuật ngữ chuyên ngành y khoa chính xác.
- Bác sĩ/y tá thường xuyên bận tay (hands-busy) hoặc mang găng tay bảo hộ.
- Mạng internet ở các khu vực như phòng mổ, chụp chiếu thường không ổn định hoặc bị cách ly.
- Cần độ trễ thấp (low latency) trong các tình huống cấp cứu.

---

## ⚖️ Tiêu chí đánh giá (Scoring Criteria)

Cuộc thi tập trung mạnh vào khả năng thực thi kỹ thuật, chia làm 3 trụ cột:

### 1. Innovation & Novelty (Đổi mới & Sáng tạo) - 25%
- Cách tiếp cận giải quyết vấn đề độc đáo.
- Tính nguyên bản và các tính năng giá trị gia tăng giúp tăng cường tính khả dụng trong thực tế (Ví dụ: thiết kế UX phù hợp cho bác sĩ, tối ưu trong môi trường ồn ào của bệnh viện, rảnh tay...).

### 2. Technical Performance (Hiệu suất Kỹ thuật) - 50%
Đây là phần quan trọng nhất, yêu cầu thiết bị phải tích hợp ML trên thiết bị (Nhận diện giọng nói + Dịch thuật + Tổng hợp giọng nói):
* **Độ chính xác (Accuracy):**
  * *Automated Metrics:* Điểm BLEU và COMET đánh giá trên tập dữ liệu benchmark độc quyền.
  * *Naturalness (Độ tự nhiên):* Đánh giá Mean Opinion Score (MOS) bởi người bản xứ về độ tự nhiên, rõ ràng, ngắt nghỉ của giọng nói máy (Synthesis).
* **Độ trễ & Hiệu năng (Efficiency & Latency):**
  * *Real-time Factor (RTF):* Hệ thống phải duy trì RTF < 1.0.
  * *Total Turnaround Latency:* Thời gian từ lúc "Kết thúc câu nói" đến lúc "Bắt đầu phát âm thanh dịch". **> 2.0 giây sẽ bị trừ điểm nặng.**
* **On-Device Constraints (Ràng buộc trên thiết bị):** 
  * Model BẮT BUỘC chạy local. **Sử dụng internet trong lúc test sẽ bị loại lập tức.**
  * *Khuyến nghị:* Nên sử dụng các models từ **Qualcomm AI Hub** đã được tối ưu (quantized, compiled) để chạy mượt trên chip Snapdragon.

### 3. Business Viability (Tính khả thi Kinh doanh) - 25%
- Khả năng sẵn sàng đưa ra thị trường của sản phẩm.
- Tính thực tế, khả năng mở rộng quy mô.
- Tính bền vững của phần cứng.

---

## 📱 Định dạng thiết bị (Device Format)

- Chấp nhận mọi định dạng: Thiết bị di động (portable), cầm tay (handheld), đeo được (wearable), hoặc thiết bị nhúng (embedded).
- Không có giới hạn hoặc bắt buộc về phân loại phần cứng. Yêu cầu giao diện thân thiện với người dùng (người lao động, bệnh nhân, bác sĩ) cần ít đào tạo nhất có thể.

---

## 🗓️ Timeline Cuộc thi (2026)

- **24/05 - 24/06:** Thời gian đăng ký.
- **Tháng 6 - Tháng 7:** Nộp Đặc tả kỹ thuật (Technical specification).
- **Tháng 9:** Nộp Sản phẩm mẫu (Prototype).
- **Tháng 10:** Thử nghiệm thực địa & Đánh giá (Field testing & evaluation).
- **Tháng 11:** Chung kết (Grand Finale) tại VNG Campus, TP. Hồ Chí Minh.
