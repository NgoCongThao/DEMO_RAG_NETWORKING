# 🤖 RAG Networking Tutor

Trợ lý ảo hỏi đáp kiến thức **Mạng Máy Tính** sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)** với truy xuất lai (Hybrid Retrieval) và trích dẫn nguồn tài liệu.

---

## ✨ Tính năng

- **Hybrid Retrieval**: Kết hợp tìm kiếm ngữ nghĩa (Vector Search – Chroma) và tìm kiếm từ khóa (BM25)
- **Trích dẫn nguồn**: Mỗi câu trả lời đều kèm tên tài liệu và số trang tham khảo
- **Từ chối thông minh**: Tự động phát hiện câu hỏi nằm ngoài phạm vi giáo trình
- **Streaming response**: Chữ AI hiện ra từng chữ theo thời gian thực (giống ChatGPT)
- **Xuất lịch sử chat**: Tải hội thoại ra file `.md` hoặc `.txt`
- **Tự điền API Key**: Đọc từ file `.env`, không cần nhập tay mỗi lần
- **Giao diện Dashboard**: Dark mode, tabs chức năng, animation AI thinking

---

## 📚 Dữ liệu sử dụng

| Tên tài liệu | Mô tả |
|---|---|
| Computer Networking: A Top-Down Approach (7th Ed.) | Giáo trình quốc tế |
| Giáo trình Mạng máy tính – PTIT | Giáo trình tiếng Việt |
| Network Fundamentals – CCNA Exploration | Chứng chỉ CCNA |
| Giáo trình Quản trị mạng | Quản trị hệ thống mạng |

---

## 🗂️ Cấu trúc dự án

```
Demo_RAG_Networking/
├── app.py                    # Ứng dụng chính (Streamlit)
├── .env                      # API Key của bạn (được gitignore, KHÔNG commit)
├── .env.example              # File mẫu — commit lên Git để hướng dẫn người dùng khác
├── data/                     # Dữ liệu PDF gốc
│   └── *.pdf
├── scripts/
│   └── build_vector_db.py    # Script tạo Chroma vector DB từ PDF
├── requirements.txt
└── README.md
```

---

## ⚙️ Cài đặt & Chạy

### 1. Clone và cài thư viện

```bash
git clone <repo-url>
cd Demo_RAG_Networking
pip install -r requirements.txt
```

### 2. Cấu hình API Key

```bash
# Sao chép file mẫu
copy .env.example .env        # Windows
cp .env.example .env          # Linux / Mac
```

Sau đó mở file `.env` và điền key của bạn (lấy tại [Google AI Studio](https://aistudio.google.com/app/apikey)):

```env
GOOGLE_API_KEY=AIzaSy...key_cua_ban
```

> ⚠️ **Không bao giờ commit file `.env`** — nó đã được thêm vào `.gitignore`.
> Nếu không tạo file `.env`, ứng dụng vẫn chạy — chỉ cần nhập key thủ công trong sidebar.

### 3. Xây dựng Vector Database

```bash
python scripts/build_vector_db.py
```

> Chạy một lần duy nhất. Kết quả lưu vào thư mục `chroma_db/` (không commit lên GitHub).

### 4. Chạy ứng dụng

```bash
streamlit run app.py
```

---

## 🔧 Công nghệ sử dụng

| Thành phần | Công nghệ |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Vector Store | ChromaDB |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` |
| Sparse Search | BM25 (rank-bm25) |
| UI | Streamlit |

---

## 📝 Lưu ý

- File `.env` chứa API Key — **không được đẩy lên GitHub** (có trong `.gitignore`).
- File `.env.example` là file mẫu an toàn — **nên commit** để hướng dẫn người clone.
- Thư mục `chroma_db/` được tạo tự động sau khi chạy `build_vector_db.py` và **không commit lên GitHub**.
- Nếu không có file `.env`, ứng dụng vẫn hoạt động bình thường, chỉ cần nhập key thủ công trong sidebar.
