# Kế hoạch Tải và Phân tích Dữ liệu (EDA) cho VietMed & Eka Medical ASR

Mục tiêu: Tải hai bộ dữ liệu ASR y tế từ Hugging Face và thực hiện phân tích khám phá dữ liệu (EDA) sâu để hiểu rõ cấu trúc, phân phối âm thanh và văn bản.

> [!WARNING]
> Bộ dữ liệu **VietMed** cực kỳ lớn (hơn 2,200 giờ âm thanh). Để tránh việc tải xuống mất hàng ngày và làm tràn ổ cứng của bạn, tôi đề xuất chỉ tải phân vùng (split) `test` hoặc giới hạn số lượng mẫu trong bước EDA này.
> Bộ **Eka Medical ASR** nhỏ hơn (~3900 mẫu), có thể tải toàn bộ một cách an toàn.

## Các bước thực hiện (Proposed Changes)

### 1. Thiết lập môi trường Python
- Tạo một môi trường ảo (virtual environment) hoặc sử dụng môi trường hiện tại.
- Cài đặt các thư viện phân tích âm thanh và dữ liệu cần thiết:
  - `datasets` (Hugging Face)
  - `pandas`, `numpy` (Xử lý dữ liệu)
  - `librosa`, `soundfile` (Xử lý âm thanh)
  - `matplotlib`, `seaborn` (Vẽ biểu đồ)

### 2. Viết kịch bản Tải Dữ liệu (Download Script)
- **Eka Dataset (`ekacare/eka-medical-asr-evaluation-dataset`)**: Tải toàn bộ cấu trúc dữ liệu.
- **VietMed Dataset (`leduckhai/VietMed`)**: Tải phân vùng `test` (hoặc một subset nhỏ bằng tính năng streaming của thư viện `datasets`) để thực hiện EDA mà không cần tải hàng trăm GB.

### 3. Thực hiện Phân tích EDA sâu (Deep EDA)
Phân tích các khía cạnh sau của cả hai tập dữ liệu:
- **Cấu trúc Dữ liệu (Data Structure):** Số lượng mẫu, các trường dữ liệu (columns), định dạng âm thanh (sampling rate).
- **Phân tích Âm thanh (Audio Analysis):** 
  - Phân phối độ dài file âm thanh (Audio Duration).
- **Phân tích Văn bản (Text Analysis):**
  - Phân phối độ dài văn bản (Word count / Character count).
  - Tần suất các từ vựng xuất hiện nhiều nhất (Top vocabularies) để xem các thuật ngữ y khoa phổ biến.

### 4. Kết xuất Báo cáo
- Sinh ra các biểu đồ (lưu dạng `.png`) vào thư mục của dự án.
- Cập nhật một file báo cáo `eda_report.md` tổng kết các phân tích.

## Open Questions

> [!IMPORTANT]
> Bạn có đồng ý với việc **chỉ tải một phần nhỏ (ví dụ 1,000 mẫu đầu tiên hoặc tập test) của bộ VietMed** để làm EDA thay vì tải toàn bộ 2,200 giờ âm thanh (có thể lên tới hàng trăm GB) không?

## Verification Plan
- Chạy script EDA độc lập.
- Kiểm tra các biểu đồ được sinh ra có hiển thị đúng phân phối âm thanh và độ dài văn bản hay không.
- Đảm bảo không gặp lỗi tràn bộ nhớ (OOM) trong quá trình xử lý audio.
