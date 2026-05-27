import streamlit as st
import os
import re
import time
import subprocess
import webbrowser
from urllib.parse import quote
from dotenv import load_dotenv

from rank_bm25 import BM25Okapi
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI

# =========================
# CONFIG UI
# =========================
st.set_page_config(
    page_title="RAG Networking Tutor",
    page_icon="🚀",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# Load .env nếu tồn tại (tự điền API key khi khởi động)
load_dotenv(os.path.join(BASE_DIR, ".env"))
_env_api_key = os.environ.get("GOOGLE_API_KEY", "")

# =========================
# CUSTOM CSS INJECTION
# =========================
st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif !important; }
/* Sidebar Logo */
.sidebar-logo { display: flex; align-items: center; gap: 12px; padding: 10px 0 20px 0; margin-bottom: 20px; }
.sidebar-logo i { font-size: 24px; color: #10a37f; }
.sidebar-logo .logo-text { font-size: 18px; font-weight: 600; }
/* Hide Streamlit elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR LAYOUT
# =========================
with st.sidebar:
    # Logo / Header
    st.markdown("""
    <div class="sidebar-logo">
        <i class="fa-solid fa-network-wired" style="color: #10a37f; margin-right: 8px;"></i>
        <div class="logo-text" style="font-size: 18px; font-weight: bold;">RAG Networking</div>
    </div>
    """, unsafe_allow_html=True)

    # Status indicator
    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="status-dot"></span>
        <span class="status-online">Hệ thống đang hoạt động</span>
    </div>
    """, unsafe_allow_html=True)

    # ── INNER TABS in Sidebar ──
    stab1, stab2 = st.tabs(["🛠️ Cài đặt", "📚 Tri thức"])

    with stab1:
        st.markdown("**🔑 Gemini API Key**")
        api_key = st.text_input(
            "API Key",
            value=_env_api_key,          # ← tự điền từ .env
            type="password",
            label_visibility="collapsed",
            placeholder="AIza..."
        )
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            if api_key == _env_api_key and _env_api_key:
                st.markdown(
                    '<span style="color:#00d4ff;font-size:11px;">⚡ Tự động tải từ file .env</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span style="color:#00ff9d;font-size:11px;">✓ API Key đã được xác nhận</span>',
                    unsafe_allow_html=True
                )

        st.divider()


        # ── EXPORT CHAT HISTORY ──
        st.markdown("**💾 Lưu lịch sử chat**")

        def build_md_export():
            """Build Markdown string from chat history."""
            lines = ["# 📋 Lịch sử hội thoại — RAG Networking Tutor\n"]
            lines.append(f"*Xuất lúc: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
            lines.append("---\n")
            for msg in st.session_state.get("messages", []):
                role_label = "🧑‍💻 **Người dùng**" if msg["role"] == "user" else "🌐 **Trợ lý RAG**"
                lines.append(f"{role_label}\n\n{msg['content']}\n\n---\n")
            return "\n".join(lines)

        def build_txt_export():
            """Build plain-text string from chat history."""
            lines = ["LỊCH SỬ HỘI THOẠI — RAG Networking Tutor"]
            lines.append(f"Xuất lúc: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append("=" * 50)
            for msg in st.session_state.get("messages", []):
                role_label = "[Người dùng]" if msg["role"] == "user" else "[Trợ lý RAG]"
                lines.append(f"{role_label}")
                lines.append(msg["content"])
                lines.append("-" * 40)
            return "\n".join(lines)

        has_history = bool(st.session_state.get("messages"))

        col_md, col_txt = st.columns(2)
        with col_md:
            st.download_button(
                label="📄 .md",
                data=build_md_export(),
                file_name="chat_history.md",
                mime="text/markdown",
                disabled=not has_history,
                use_container_width=True,
            )
        with col_txt:
            st.download_button(
                label="📝 .txt",
                data=build_txt_export(),
                file_name="chat_history.txt",
                mime="text/plain",
                disabled=not has_history,
                use_container_width=True,
            )

        if not has_history:
            st.caption("_Chưa có cuộc hội thoại nào._")

        if has_history:
            if st.button("🗑️ Xóa lịch sử chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    with stab2:
        st.markdown("**📖 Cơ sở tri thức**")
        st.markdown("""
        <div class="book-card"><span class="book-icon">📗</span> Thiết Kế Mạng</div>
        <div class="book-card"><span class="book-icon">📘</span> CCNA (Cisco)</div>
        <div class="book-card"><span class="book-icon">📙</span> Computer Networking</div>
        """, unsafe_allow_html=True)

        st.divider()

        if st.button("🔄 Tải lại Database"):
            with st.spinner("Đang tải lại..."):
                st.cache_resource.clear()
                time.sleep(1)
                st.success("✅ Database đã được tải lại!")
                st.rerun()

# =========================
# LOAD RESOURCES  (logic unchanged)
# =========================
@st.cache_resource
def load_resources():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    vector_db = Chroma(
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    data = vector_db.get()

    return vector_db, data["documents"], data["metadatas"]

# =========================
# CLEAN TEXT  (logic unchanged)
# =========================
def clean_text(text):

    return re.sub(r'[^\w\s/]', '', text.lower())

# =========================
# INIT DATA  (logic unchanged)
# =========================
@st.cache_resource
def build_bm25(documents):
    tokenized_corpus = [clean_text(doc).split() for doc in documents]
    return BM25Okapi(tokenized_corpus)

try:

    vector_db, documents, metadatas = load_resources()

    bm25 = build_bm25(documents)

except Exception as e:

    st.error(f"Lỗi tải dữ liệu: {e}")

    st.stop()

# =========================
# MAIN PAGE HEADER
# =========================
st.markdown("""
<div class="page-header">
    <i class="fa-solid fa-robot" style="font-size:36px; color:#10a37f; margin-right: 15px;"></i>
    <div style="display: inline-block; vertical-align: middle;">
        <h1 style="margin: 0;">Trợ lý ảo Mạng Máy Tính</h1>
        <p class="subtitle" style="margin: 0; color: gray;">RAG · Gemini 2.5 Flash</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# CHAT HISTORY
# =========================
if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    avatar = "user" if message["role"] == "user" else "assistant"
    with st.chat_message(message["role"], avatar=avatar):

        st.markdown(message["content"])
        
        if message.get("citations"):
            with st.expander("Xem nguồn trích dẫn từ giáo trình"):
                for idx, d in enumerate(message["citations"]):
                    src = d['metadata'].get('source', 'Không rõ')
                    page = d['metadata'].get('page', '?')
                    preview = d['content'][:400]
                    st.success(
                        f"**📖 [{idx+1}] {src}** — Trang {page}\n\n"
                        f"{preview}\u2026"
                    )

# =========================
# USER INPUT
# =========================
if prompt := st.chat_input("Nhập câu hỏi về mạng máy tính… (Enter để gửi)"):

    if not api_key:

        st.warning("⚠️ Vui lòng nhập Gemini API Key ở thanh bên trái (Tab 🛠️ Cài đặt).")

        st.stop()

    # =========================
    # USER MESSAGE
    # =========================
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user", avatar="user"):

        st.markdown(prompt)

    # =========================
    # ASSISTANT
    # =========================
    with st.chat_message("assistant", avatar="assistant"):

        # Thinking animation placeholder
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="thinking-animation">
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <div class="thinking-dot"></div>
            <span style="margin-left:4px;">AI đang suy nghĩ…</span>
        </div>
        """, unsafe_allow_html=True)

        with st.status(
            "🔍 Đang lục tìm tài liệu...",
            expanded=False
        ) as status:

            time.sleep(1)

            # =========================
            # 1. QUERY ANALYSIS & VECTOR SEARCH
            # =========================
            normalized_prompt = prompt.strip().lower()
            top_k = 5

            vector_results_raw = vector_db.similarity_search_with_score(
                normalized_prompt,
                k=top_k
            )
            # similarity_search_with_score returns a list of tuples (Document, score)
            vector_results = [doc for doc, score in vector_results_raw]

            # =========================
            # 2. BM25 SEARCH
            # =========================
            tokenized_query = clean_text(normalized_prompt).split()

            scores = bm25.get_scores(tokenized_query)

            if len(scores) == 0 or max(scores) < 1:
                top_bm25_idx = []
            else:
                top_bm25_idx = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )[:top_k]

            # =========================
            # 3. COMBINE RESULTS  (logic unchanged)
            # =========================
            combined = []

            for doc in vector_results:

                combined.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            for i in top_bm25_idx:

                combined.append({
                    "content": documents[i],
                    "metadata": metadatas[i]
                })

            # =========================
            # 4. REMOVE DUPLICATE CONTENT  (logic unchanged)
            # =========================
            seen_content = set()

            final_results = []

            for d in combined:

                if d["content"] not in seen_content:

                    final_results.append(d)

                    seen_content.add(d["content"])

            # =========================
            # 5. RERANKING  (logic unchanged)
            # =========================
            stop_words = [
                "là",
                "gì",
                "của",
                "và",
                "những",
                "các",
                "cho",
                "được"
            ]

            tu_khoa = [
                w for w in tokenized_query
                if w not in stop_words
            ]

            # Fallback nếu query toàn stop words
            if not tu_khoa:
                tu_khoa = tokenized_query

            if tu_khoa:
                # Sắp xếp dựa trên:
                # 1. Số lượng từ khóa độc lập xuất hiện trong document (càng nhiều từ khớp càng tốt)
                # 2. Tổng số lần xuất hiện của tất cả từ khóa
                final_results.sort(
                    key=lambda x: (
                        sum(1 for w in tu_khoa if w in clean_text(x["content"])),
                        sum(clean_text(x["content"]).count(w) for w in tu_khoa)
                    ),
                    reverse=True
                )

            # =========================
            # 6. REMOVE DUPLICATE PAGES  (logic unchanged)
            # =========================
            filtered_docs = []

            seen_pages = set()

            for d in final_results:

                page_key = (
                    d["metadata"].get("source"),
                    d["metadata"].get("page")
                )

                if page_key not in seen_pages:

                    filtered_docs.append(d)

                    seen_pages.add(page_key)

                if len(filtered_docs) == 5:
                    break

            top_docs = filtered_docs

            # =========================
            # 7. REFUSAL CHECK  (logic unchanged)
            # =========================
            if (
                not top_docs
                or len(top_docs[0]["content"]) < 50
            ):

                thinking_placeholder.empty()
                st.warning(
                    "⚠️ Không tìm thấy thông tin phù hợp trong tài liệu.\n\n"
                    "*(Một số tài liệu có thể là PDF scan ảnh và không thể trích xuất text đầy đủ)*"
                )

                st.stop()

            status.update(
                label="🧠 Đang nhờ AI đọc và phân tích độ phù hợp...",
            )

        # =========================
        # BUILD CONTEXT  (logic unchanged)
        # =========================
        context = "\n\n====================\n\n".join(
            [
                f"TÀI LIỆU: {d['metadata'].get('source')} "
                f"| TRANG: {d['metadata'].get('page')}\n\n"
                f"{d['content']}"
                for d in top_docs
            ]
        )

        # =========================
        # SYSTEM PROMPT  (logic unchanged)
        # =========================
        system_prompt = f"""
Bạn là trợ lý AI chuyên ngành Mạng Máy Tính.

NHIỆM VỤ:
- Bạn phải đọc NGỮ CẢNH được cung cấp. Nếu ngữ cảnh chứa thông tin trả lời được câu hỏi, hãy tổng hợp lại.
- Tùyệt đối không tự suy diễn kiến thức bên ngoài internet.

QUY TẮC TRẢ LỜI:
1. TRƯỜNG HỢP CÓ THÔNG TIN TRONG NGỮ CẢNH: 
- Luôn trả lời bằng tiếng Việt
- Trả lời chi tiết, đầy đủ và cặn kẽ, tổng hợp thông tin từ nhiều nguồn được cung cấp. Tránh trả lời quá ngắn gọn.
- Chỉ sử dụng thông tin có trong NGỮ CẢNH.
- Không suy đoán hoặc tự bổ sung kiến thức ngoài tài liệu.
- Nếu người dùng yêu cầu "so sánh", hãy trình bày theo từng tiêu chí rõ ràng (ví dụ: dùng bảng hoặc danh sách liệt kê).
- Trình bày mạch lạc, kết hợp các ý chính từ tài liệu một cách toàn diện thay vì chỉ sao chép nguyên văn.
- Giải thích theo phong cách giảng dạy cho sinh viên CNTT.
- Nếu xuất hiện thuật ngữ viết tắt, hãy giải thích đầy đủ tên thuật ngữ.
- Khi sử dụng thông tin từ tài liệu nào, hãy nhắc tên tài liệu và số trang ngay trong đoạn trả lời (ví dụ: Theo tài liệu CCNA (Trang 45)...).
- Cuối câu trả lời phải có mục: "Nguồn tham khảo: [Tên tài liệu], Trang [số trang]" để người dùng dễ dàng tra cứu.

2. TRƯỜNG HỢP KHÔNG CÓ THÔNG TIN (Ngữ cảnh không liên quan đến câu hỏi):
- CHỈ ĐƯỢC PHÉP trả lời đúng 1 câu duy nhất sau đây, không thêm bất kỳ từ nào khác:
"Rất tiếc, tài liệu hiện tại không đề cập đến nội dung này."

====================
NGỮ CẢNH:
{context}
====================

CÂU HỎI:
{prompt}
"""

        # =========================
        # GENERATION — Streaming
        # =========================
        try:

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0.3
            )

            # Clear thinking animation before streaming starts
            thinking_placeholder.empty()

            # Stream response token by token
            def token_stream():
                for chunk in llm.stream(system_prompt):
                    yield chunk.content

            full_response = st.write_stream(token_stream())

            if "Rất tiếc" in full_response:
                status.update(
                    label="❌ Câu hỏi nằm ngoài phạm vi giáo trình!",
                    state="error"
                )
            else:
                status.update(
                    label="✅ Đã tổng hợp xong câu trả lời từ tài liệu!",
                    state="complete"
                )

            # =========================
            # CITATION & DISPLAY LOGIC
            # =========================
            if "Rất tiếc" in full_response:
                # Nếu AI từ chối, chỉ hiện câu từ chối, KHÔNG hiện nguồn trích dẫn
                with st.expander("Xem nguồn trích dẫn từ giáo trình"):
                    st.info("Hệ thống đã truy xuất tài liệu nhưng không tìm thấy đoạn văn bản nào có ngữ nghĩa phù hợp với câu hỏi của bạn.")
            else:


                # Nếu AI trả lời bình thường, hiện danh sách trích dẫn dạng card
                with st.expander("Xem nguồn trích dẫn từ giáo trình"):
                    for idx, d in enumerate(top_docs):
                        src  = d['metadata'].get('source', 'Không rõ')
                        page = d['metadata'].get('page', '?')
                        preview = d['content'][:400]

                        # Hiện card trích dẫn
                        st.success(
                            f"**📖 [{idx+1}] {src}** — Trang {page}\n\n"
                            f"{preview}\u2026"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citations": [] if "Rất tiếc" in full_response else top_docs
            })

        except Exception as e:

            thinking_placeholder.empty()
            st.error(f"Lỗi AI: {e}")