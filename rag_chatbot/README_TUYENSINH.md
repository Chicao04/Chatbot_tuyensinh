# RAG Chatbot Tuyển sinh (chuyennganh/diem_chuan/thacsi_tiensi) - Gemini

## 1) Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# sửa GEMINI_API_KEY trong .env
```

## 2) Đầu vào dữ liệu (.txt)
Hệ thống chỉ index file **.txt** theo đúng thư mục:

```
input/
  chuyennganh/
    *.txt
  diem_chuan/
    *.txt
  thacsi_tiensi/
    *.txt
```

## 3) Build index
```bash
python build_index_tuyensinh.py
```

## 4) Chat
```bash
python main_tuyensinh.py
```

## 5) Lưu ý về thư viện Gemini
Khuyến nghị cài:
```bash
pip install -U google-genai
```
Nếu bạn dùng bản cũ:
```bash
pip install -U google-generativeai
```
