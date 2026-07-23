<div align="center">
  <h1>🏥 Báo cáo Hệ sinh thái Dataset Y tế (OneVoice Project)</h1>
  <p><i>Phân tích, Đánh giá, Khám phá dữ liệu (EDA) và Chiến lược Merge</i></p>
</div>

> [!NOTE]
> Báo cáo này được tổng hợp dựa trên yêu cầu dự án và **kết quả EDA từ Google Colab**. Các dataset được phân tích chi tiết về loại dữ liệu, định dạng (format) lưu trữ chuẩn để dễ dàng merge với các team member khác (Thái & Khoa), cùng các thông số thống kê cụ thể.

---

## 📊 1. Dữ liệu Dịch máy (Machine Translation & LLM)
*Mục đích: Đào tạo mô hình dịch thuật Việt ↔ Anh và khả năng tư vấn y tế.*

### 1.1. hungsvdut2k2/vietnamese-medical-chat-data
| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Loại Data** | `Text` (Hội thoại / Chat) |
| **Quy mô (EDA)** | **46,479** mẫu (Không có dữ liệu rỗng/thiếu) |
| **Mục đích** | Bổ sung tính giao tiếp tự nhiên, văn phong hỏi đáp thực tế. |
| **Khả năng Merge** | Kết hợp cực tốt với bộ QA của Khoa (MedQuad) để "bản địa hóa" (localize) câu hỏi cho thị trường Việt Nam. |
| **Format Chuẩn** | Định dạng ChatML / OpenAI Messages. |

**Đề xuất cấu trúc lưu trữ (Format):**
```json
{
  "messages": [
    {"role": "system", "content": "Bạn là một trợ lý y tế ảo của hệ thống OneVoice... "},
    {"role": "user", "content": "Bác sĩ ơi, tôi hay bị đau nhức xương khớp mỗi khi trời lạnh..."},
    {"role": "assistant", "content": "Chào bạn, triệu chứng đau xương khớp khi chuyển mùa có thể do..."}
  ]
}
```

### 1.2. FutureBeeAI English-Vietnamese Medical
| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Loại Data** | `Text` (Song ngữ Parallel) |
| **Quy mô** | > 50,000 cặp câu song ngữ. |
| **Mục đích** | Dataset nền tảng cho dịch thuật (Core Translation). |
| **Khả năng Merge** | Hoàn toàn độc lập với **MedEV của Thái**. Dùng làm bộ test/validation hoặc gộp chung với MedEV theo tỷ lệ (60% MedEV - 40% FutureBeeAI) để giảm bớt tính hàn lâm. |
| **Format Chuẩn** | Cấu trúc Instruction-based (Alpaca format). |

**Đề xuất cấu trúc lưu trữ (Format):**
```json
{
  "instruction": "Dịch đoạn văn bản y tế sau từ Tiếng Anh sang Tiếng Việt.",
  "input": "Patient exhibits severe anaphylactic shock symptoms.",
  "output": "Bệnh nhân có biểu hiện sốc phản vệ nghiêm trọng."
}
```

### 1.3. MedEV (Dữ liệu do Thái phụ trách)
> [!WARNING]
> Thái chịu trách nhiệm bộ này (~360,000 cặp câu). Bạn không cần tải lại, chỉ cần cung cấp mã hash (MD5) của tập dữ liệu của bạn để Thái chạy Deduplication (khử trùng lặp) trước khi merge.

---

## 🎙️ 2. Dữ liệu Nhận diện Giọng nói (ASR)
*Mục đích: Xử lý giọng nói tiếng Việt có dấu, Code-switching, và tiếng Anh của bác sĩ nước ngoài.*

### 2.1. tensorxt/ViMedCSS (Code-switching Y tế)
| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Loại Data** | `Audio` + `Text Transcript` (Tiếng Việt pha Tiếng Anh) |
| **Quy mô (EDA)** | **11,832** mẫu (split: train). |
| **Đặc điểm** | Rất quan trọng để mô hình nghe được bác sĩ Việt Nam nói chèn tiếng Anh (vd: "làm test", "bị virus"). |
| **Format Chuẩn** | HuggingFace Audio Format (`Array` + `Sampling Rate` + `Transcript`). |

**Đề xuất cấu trúc metadata (Format):**
```json
{
  "audio_path": "wavs/vimedcss_0001.wav",
  "sampling_rate": 16000,
  "transcript": "Bệnh nhân cần làm test PCR ngay lập tức.",
  "language_tag": "vi-code-switch"
}
```

### 2.2. leduckhai/VietMed (Vietnamese ASR)
| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Loại Data** | `Audio` + `Text Transcript` |
| **Quy mô (EDA)** | **3,437** mẫu (split: test). Bộ full chứa 16h labeled + 1000h unlabeled. |
| **Mục đích** | Chuẩn mực nhất cho ASR tiếng Việt, đa dạng vùng miền (Bắc/Trung/Nam). |
| **Format Chuẩn** | Tương tự ViMedCSS. Nên thêm thẻ vùng miền nếu có. |

### 2.3. ekacare/eka-medical-asr-evaluation-dataset
| Thuộc tính | Chi tiết |
| :--- | :--- |
| **Loại Data** | `Audio` + `Text Transcript` (Tiếng Anh) |
| **Quy mô (EDA)** | **3,619** mẫu (split: test). |
| **Mục đích** | Tối ưu nhận diện thuật ngữ y khoa bằng tiếng Anh. |
| **Khả năng Merge** | Gộp chung với bộ của Kaggle (Medical Speech). Khi lưu cần gắn cờ `language_tag: "en-medical"`. |

---

## 🛠️ 3. Quy trình chuẩn bị Data & Khử trùng lặp (Merge Strategy)

Để hệ thống dữ liệu giữa bạn, Thái và Khoa đồng nhất, hãy tuân thủ quy trình sau:

> [!IMPORTANT]
> **Quy tắc Vàng khi Merge:** Tất cả dữ liệu TEXT phải chuyển về định dạng JSON Lines (`.jsonl`) để dễ dàng đọc bằng Pandas/HuggingFace Dataset mà không tràn RAM.

1.  **Bước 1: Chuẩn hóa Schema (Schema Normalization)**
    *   Thống nhất 1 schema JSON chung. Ví dụ, cho tác vụ Text/Chat, sử dụng schema `role/content` của OpenAI. Cho tác vụ ASR, sử dụng schema `audio_path/transcript/sampling_rate`.
2.  **Bước 2: Deduplication (Xử lý trùng lặp chéo)**
    *   Tạo ra một file mã Hash (ví dụ: MinHash hoặc SHA-256) cho từng câu transcript của bạn.
    *   Gửi file Hash này cho Khoa và Thái để họ so khớp với database của họ. Nếu trùng Hash, xóa bản ghi ở 1 bên.
3.  **Bước 3: Tagging (Gắn nhãn nguồn)**
    *   Thêm metadata `"source": "hungsvdut_chat"` hoặc `"source": "vimedcss"` để khi huấn luyện, model biết được ngữ cảnh (prompt context) của data.

---

## 🔍 4. Hướng Tìm kiếm Dữ liệu Bổ sung (Innovation Directions)

Để tăng tính đột phá (Novelty) cho thiết bị Edge AI của OneVoice, bạn nên tìm kiếm bổ sung:

1.  **Noise Augmentation Data (Dữ liệu tiếng ồn):**
    *   *Tại sao:* Thiết bị dùng ở bệnh viện sẽ rất ồn. Model train bằng Audio phòng thu sẽ thất bại.
    *   *Keyword:* `ESC-50 dataset hospital noise`, `AudioSet ambulance siren`, `clinical background babble noise`.
    *   *Action:* Mix các âm thanh này đè lên các audio chuẩn của *VietMed* với tỷ lệ SNR (Signal-to-Noise Ratio) từ 5dB đến 15dB.
2.  **Disfluency & Spontaneous Speech (Nói ngắc ngứ):**
    *   *Tại sao:* Bác sĩ khi cấp cứu thường nói vấp (vd: "ờ... lấy cho tôi cái... cái nẹp").
    *   *Keyword:* `Medical spontaneous speech dataset`, `ASR disfluency corpus`.
3.  **Vietnamese G2P for English Drugs (Phát âm lóng tên thuốc):**
    *   *Tại sao:* Bệnh nhân hay đọc "Panadol" thành "Pa-na-đôn", "Paracetamol" thành "Pa-ra-xê-ta-môn".
    *   *Action:* Tự tạo một bộ từ điển `grapheme-to-phoneme (G2P)` tự chế map các từ này về tiếng Việt để inject vào mô hình ASR.
