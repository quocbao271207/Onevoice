# Onevoice — EDA dữ liệu ASR y tế tiếng Việt

Tải và phân tích khám phá dữ liệu (EDA) hai bộ dữ liệu nhận dạng tiếng nói (ASR) y tế tiếng Việt: **VietMed** và **Eka Medical ASR** (từ Hugging Face).

## Cấu trúc
- `notebooks/` — notebook EDA
- `src/` — mã nguồn
- `eda_images/` — biểu đồ EDA
- `docs/`, `implementation_plan.md` — kế hoạch & tài liệu

## Môi trường
```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

> ⚠️ Dữ liệu ASR **không** lưu trong repo — tải lại từ Hugging Face. VietMed rất lớn (>2200 giờ âm thanh), nên chỉ tải split `test` hoặc giới hạn số mẫu khi EDA.
