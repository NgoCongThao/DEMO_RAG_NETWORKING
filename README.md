# 🤖 RAG Networking Tutor

Trợ lý ảo hỏi đáp kiến thức **Mạng Máy Tính** sử dụng kỹ thuật **Retrieval-Augmented Generation (RAG)** với truy xuất lai (Hybrid Retrieval) và trích dẫn nguồn tài liệu.

---

## ✨ Tính năng

- **Hybrid Retrieval**: Kết hợp tìm kiếm ngữ nghĩa (Vector Search – ChromaDB) và tìm kiếm từ khóa (BM25)
- **Smart Reranking**: Tái xếp hạng tài liệu dựa trên điểm keyword và ngữ nghĩa chuyên ngành
- **Trích dẫn nguồn**: Mỗi câu trả lời đều kèm tên tài liệu và số trang tham khảo
- **Từ chối thông minh**: Tự động phát hiện câu hỏi nằm ngoài phạm vi giáo trình
- **Streaming response**: Câu trả lời hiển thị từng chữ theo thời gian thực
- **Xuất lịch sử chat**: Tải hội thoại ra file `.md` hoặc `.txt`
- **Tự điền API Key**: Đọc tự động từ file `.env`, không cần nhập tay mỗi lần
- **Debug Mode**: Xem các đoạn context thô mà LLM đã từ chối

---

## 🖥️ Yêu cầu hệ thống

| Yêu cầu | Chi tiết |
|---|---|
| Python | **3.10 – 3.12** (khuyến nghị 3.11) |
| RAM | Tối thiểu **8 GB** (embedding model cần ~500 MB) |
| Dung lượng | ~2 GB (model + database) |
| Kết nối | Cần Internet để gọi Gemini API |

---

## 📚 Dữ liệu sử dụng

| Tên tài liệu | Loại |
|---|---|
| Computer Networking: A Top-Down Approach (7th Ed.) | Giáo trình quốc tế |
| Giáo trình Mạng Máy Tính – Ngô Bá Hùng & Phạm Thế Phi | Giáo trình tiếng Việt |
| Network Fundamentals – CCNA Exploration Companion Guide | Tài liệu chứng chỉ CCNA |
| CCNP Enterprise Design ENSLD 300-420 Official Cert Guide | Tài liệu chứng chỉ CCNP |
| Giáo trình Quản trị Mạng | Quản trị hệ thống mạng |

---

## 🗂️ Cấu trúc dự án

```
Demo_RAG_Networking/
├── app.py                    # Ứng dụng chính (Streamlit)
├── requirements.txt          # Danh sách thư viện
├── .env                      # API Key cá nhân (KHÔNG commit lên Git)
├── .env.example              # File mẫu cấu hình (an toàn để commit)
├── README.md
├── data/                     # Thư mục chứa file PDF nguồn
│   └── *.pdf
├── scripts/
│   └── build_vector_db.py    # Script xây dựng Chroma vector database
└── chroma_db/                # Database được tạo tự động (KHÔNG commit lên Git)
```

---

## ⚙️ Hướng dẫn cài đặt & chạy

### Bước 1 — Tải source code

```bash
git clone <repo-url>
cd Demo_RAG_Networking
```

Hoặc giải nén file `.zip` nếu được cung cấp dưới dạng archive.

---

### Bước 2 — Tạo môi trường ảo và cài thư viện

**Khuyến nghị dùng môi trường ảo để tránh xung đột thư viện:**

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt (Windows)
.venv\Scripts\activate

# Kích hoạt (Linux / macOS)
source .venv/bin/activate

# Cài đặt thư viện
pip install -r requirements.txt
```

> ⏱️ Lần đầu cài đặt có thể mất 3–10 phút do tải model embedding (~500 MB).

---

### Bước 3 — Cấu hình Gemini API Key

**3.1. Lấy API Key miễn phí tại:** [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

**3.2. Tạo file `.env` từ file mẫu:**

```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

**3.3. Mở file `.env` và điền key của bạn:**

```env
GOOGLE_API_KEY=AIzaSy...key_cua_ban
```

> ⚠️ **Không bao giờ commit file `.env`** lên Git — nó đã được thêm vào `.gitignore`.  
> Nếu không tạo file `.env`, ứng dụng vẫn hoạt động bình thường — chỉ cần nhập key thủ công trong sidebar khi chạy.

---

### Bước 4 — Đặt file PDF vào thư mục `data/`

Sao chép các file PDF giáo trình vào thư mục `data/`:

```
Demo_RAG_Networking/
└── data/
    ├── Computer Networking A Top-Down Approach, 7th Edition.pdf
    ├── Giáo Trình Mạng Máy Tính (Ngô Bá Hùng Phạm Thế Phi).pdf
    ├── Network Fundamentals CCNA Exploration Companion Guide.pdf
    ├── CCNP Enterprise Design ENSLD 300-420 Official Cert Guide.pdf
    └── giao-trinh-quan-tri-mang.pdf
```

---

### Bước 5 — Xây dựng Vector Database

```bash
python scripts/build_vector_db.py
```

Script sẽ tự động:
1. Đọc tất cả file PDF trong thư mục `data/`
2. Trích xuất text từng trang
3. Chia nhỏ thành các đoạn (chunk size: 500 ký tự, overlap: 100)
4. Tạo vector embeddings bằng model `paraphrase-multilingual-MiniLM-L12-v2`
5. Lưu toàn bộ vào thư mục `chroma_db/`

> ⏱️ Quá trình này có thể mất **5–15 phút** tùy số lượng PDF và cấu hình máy.  
> Chỉ cần chạy **một lần duy nhất**. Nếu muốn cập nhật tài liệu, chạy lại script này.

Kết quả mong đợi khi thành công:
```
[DONE] Build database thành công!
```

---

### Bước 6 — Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ tự động mở trình duyệt tại:
- **Local:** `http://localhost:8501`
- **Network:** `http://<IP-máy-bạn>:8501`

---

## 🔧 Công nghệ sử dụng

| Thành phần | Công nghệ | Phiên bản |
|---|---|---|
| LLM | Google Gemini 2.5 Flash | API |
| Vector Store | ChromaDB | 1.5.8 |
| Embeddings | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace |
| Sparse Search | BM25 (rank-bm25) | 0.2.2 |
| PDF Parsing | PyMuPDF (fitz) | 1.27.2 |
| UI Framework | Streamlit | 1.57.0 |

---

## 🛠️ Xử lý sự cố thường gặp

### ❌ Lỗi `ModuleNotFoundError`
```bash
# Kiểm tra môi trường ảo đã được kích hoạt chưa
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux/macOS

# Cài lại thư viện
pip install -r requirements.txt
```

### ❌ Lỗi `GOOGLE_API_KEY not found` hoặc API Key không hợp lệ
- Kiểm tra file `.env` tồn tại và đúng định dạng: `GOOGLE_API_KEY=AIza...`
- Hoặc nhập key trực tiếp vào ô **🔑 Gemini API Key** trong sidebar của ứng dụng

### ❌ Lỗi khi build database (`build_vector_db.py`)
- Đảm bảo thư mục `data/` tồn tại và chứa ít nhất một file `.pdf`
- Kiểm tra file PDF không bị hỏng và có thể mở được
- Nếu PDF là dạng **scan ảnh** (không có text), script sẽ báo `[WARNING]` và bỏ qua

### ❌ Ứng dụng trả lời "Rất tiếc, tài liệu không đề cập..."
- Câu hỏi nằm ngoài phạm vi các tài liệu đã nạp
- Thử bật **Debug Mode** trong sidebar để xem các đoạn context đã được truy xuất

### ❌ `streamlit` không được nhận diện
```bash
# Dùng lệnh thay thế
python -m streamlit run app.py
```

---

## 📝 Lưu ý quan trọng

- File `.env` chứa API Key — **không được đẩy lên GitHub**.
- Thư mục `chroma_db/` được tạo tự động và **không commit lên GitHub** (đã có trong `.gitignore`).
- File `.env.example` là file mẫu an toàn — **nên giữ lại** để hướng dẫn người khác.
- Mỗi lần thêm tài liệu PDF mới vào `data/`, cần chạy lại `build_vector_db.py` để cập nhật database.
