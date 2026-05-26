# 🤖 RAG Networking Tutor

Trợ lý ảo hỏi đáp kiến thức **Mạng Máy Tính** sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)** với truy xuất lai (Hybrid Retrieval) và trích dẫn nguồn tài liệu.

---

## ✨ Tính năng

- **Hybrid Retrieval**: Kết hợp tìm kiếm ngữ nghĩa (Vector Search – Chroma) và tìm kiếm từ khóa (BM25)
- **Trích dẫn nguồn**: Mỗi câu trả lời đều kèm tên tài liệu và số trang tham khảo
- **Từ chối thông minh**: Tự động phát hiện câu hỏi nằm ngoài phạm vi giáo trình
- **Giao diện chat**: Xây dựng bằng Streamlit, hỗ trợ lịch sử hội thoại

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
├── data/                     # Dữ liệu PDF gốc
│   ├── *.pdf
├── scripts/
│   └── build_vector_db.py    # Script tạo Chroma vector DB từ PDF
└── README.md
```

---

## ⚙️ Cài đặt & Chạy

### 1. Cài thư viện

```bash
pip install streamlit langchain langchain-community langchain-google-genai \
            chromadb sentence-transformers rank-bm25 pypdf
```

### 2. Xây dựng Vector Database

```bash
python scripts/build_vector_db.py
```

> Chạy một lần duy nhất. Kết quả lưu vào thư mục `chroma_db/` (không commit lên GitHub).

### 3. Chạy ứng dụng

```bash
streamlit run app.py
```

### 4. Nhập Gemini API Key

Mở sidebar → nhập **Gemini API Key** của bạn (lấy tại [Google AI Studio](https://aistudio.google.com/)).

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

- Thư mục `chroma_db/` được tạo tự động sau khi chạy `build_vector_db.py` và **không được đẩy lên GitHub**.
- Cần có Gemini API Key để sử dụng.
