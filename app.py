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

load_dotenv(os.path.join(BASE_DIR, ".env"))
_env_api_key = os.environ.get("GOOGLE_API_KEY", "")

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, .stApp { font-family: 'Inter', sans-serif !important; }
.sidebar-logo { display: flex; align-items: center; gap: 12px; padding: 10px 0 20px 0; margin-bottom: 20px; }
.sidebar-logo i { font-size: 24px; color: #10a37f; }
.sidebar-logo .logo-text { font-size: 18px; font-weight: 600; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <i class="fa-solid fa-network-wired" style="color: #10a37f; margin-right: 8px;"></i>
        <div class="logo-text" style="font-size: 18px; font-weight: bold;">RAG Networking</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:16px;">
        <span class="status-dot"></span>
        <span class="status-online">Hệ thống đang hoạt động</span>
    </div>
    """, unsafe_allow_html=True)

    retrieval_mode = st.selectbox(
        "⚙️ Chế độ Retrieval",
        [
            "Hybrid Retrieval",
            "Vector Search Only",
            "BM25 Only"
        ],
        help="Chọn chiến lược tìm kiếm tài liệu để so sánh độ chính xác."
    )

    debug_mode = st.toggle("🐞 Debug Mode (Hiển thị context từ chối)", value=False)

    stab1, stab2 = st.tabs(["🛠️ Cài đặt", "📚 Tri thức"])

    with stab1:
        st.markdown("**🔑 Gemini API Key**")
        api_key = st.text_input(
            "API Key",
            value=_env_api_key,
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
            """Tạo chuỗi Markdown từ lịch sử hội thoại."""
            lines = ["# 📋 Lịch sử hội thoại — RAG Networking Tutor\n"]
            lines.append(f"*Xuất lúc: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")
            lines.append("---\n")
            for msg in st.session_state.get("messages", []):
                role_label = "🧑‍💻 **Người dùng**" if msg["role"] == "user" else "🌐 **Trợ lý RAG**"
                lines.append(f"{role_label}\n\n{msg['content']}\n\n---\n")
            return "\n".join(lines)

        def build_txt_export():
            """Tạo chuỗi plain-text từ lịch sử hội thoại."""
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
# LOAD RESOURCES
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
# CLEAN TEXT
# =========================
def clean_text(text):
    return re.sub(r'[^\w\s/]', '', text.lower())

# =========================
# BUILD BM25 INDEX
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
                if "Rất tiếc" in message["content"]:
                    st.info("Hệ thống đã truy xuất tài liệu nhưng không tìm thấy đoạn văn bản nào có ngữ nghĩa phù hợp với câu hỏi của bạn.")
                    if debug_mode:
                        st.divider()
                        st.markdown("**[DEBUG MODE] Các ngữ cảnh thô bị LLM từ chối:**")
                        for idx, d in enumerate(message["citations"]):
                            src = d['metadata'].get('source', 'Không rõ')
                            page = d['metadata'].get('page', '?')
                            preview = d['content'][:400]
                            st.warning(
                                f"**📖 [{idx+1}] {src}** — Trang {page}\n\n"
                                f"{preview}\u2026"
                            )
                else:
                    for idx, d in enumerate(message["citations"][:3]):
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

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user", avatar="user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="assistant"):

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

            # Loại bỏ stop words trước khi tokenize query
            stop_words = [
                "là", "gì", "của", "và",
                "những", "các", "cho",
                "được", "như", "thế",
                "nào", "tại", "sao"
            ]

            tokenized_query = clean_text(prompt).split()

            tu_khoa_loi = [
                w for w in tokenized_query
                if w not in stop_words
            ]

            if not tu_khoa_loi:
                tu_khoa_loi = tokenized_query

            combined = []

            # =========================
            # VECTOR SEARCH ONLY
            # =========================
            if retrieval_mode == "Vector Search Only":

                vector_results = vector_db.similarity_search(
                    prompt,
                    k=3
                )

                for doc in vector_results:
                    combined.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })

            # =========================
            # BM25 ONLY
            # =========================
            elif retrieval_mode == "BM25 Only":

                scores = bm25.get_scores(tu_khoa_loi)

                top_bm25_idx = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )[:3]

                for i in top_bm25_idx:
                    combined.append({
                        "content": documents[i],
                        "metadata": metadatas[i]
                    })

            # =========================
            # HYBRID RETRIEVAL
            # =========================
            else:

                vector_results = vector_db.similarity_search(
                    prompt,
                    k=6
                )

                for doc in vector_results:
                    combined.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata,
                        "source_type": "vector",
                        "score_bonus": 2
                    })

                # Chỉ dùng từ kỹ thuật để tránh nhiễu khi tìm kiếm BM25
                technical_keywords = [
                    w for w in tu_khoa_loi
                    if (
                        w.isupper()
                        or "/" in w
                        or len(w) >= 4
                    )
                ]
                bm25_query = technical_keywords if technical_keywords else tu_khoa_loi
                scores = bm25.get_scores(bm25_query)

                top_bm25_idx = sorted(
                    range(len(scores)),
                    key=lambda i: scores[i],
                    reverse=True
                )[:4]

                for i in top_bm25_idx:
                    combined.append({
                        "content": documents[i],
                        "metadata": metadatas[i],
                        "source_type": "bm25",
                        "score_bonus": 0
                    })

            # =========================
            # LOẠI BỎ TRÙNG LẶP
            # =========================
            seen = set()
            final_results = []

            for d in combined:
                if d["content"] not in seen:
                    final_results.append(d)
                    seen.add(d["content"])

            # =========================
            # SMART HYBRID RERANKING
            # =========================
            if retrieval_mode == "Hybrid Retrieval" and tu_khoa_loi:

                def keyword_match_score(content, keywords):
                    """
                    Tính điểm keyword matching:
                    - Exact match  : +3
                    - Partial match: +1
                    """
                    content = clean_text(content)
                    score = 0
                    for w in keywords:
                        if w in content:
                            score += 3
                        elif any(
                            w in token or token in w
                            for token in content.split()
                        ):
                            score += 1
                    return score

                for d in final_results:

                    clean_content = clean_text(d["content"])

                    distinct_match = sum(
                        1 for w in tu_khoa_loi
                        if w in clean_content
                    )

                    exact_match = 0
                    partial_match = 0
                    for w in tu_khoa_loi:
                        if w in clean_content:
                            exact_match += 1
                        elif any(
                            w in token
                            for token in clean_content.split()
                        ):
                            partial_match += 1

                    # Giới hạn mật độ keyword tối đa 8 để tránh overfitting
                    density = min(
                        sum(clean_content.count(w) for w in tu_khoa_loi),
                        8
                    )

                    source_bonus = 2 if d.get("source_type") == "vector" else 0

                    # Bonus ngữ nghĩa dựa trên từ khoá chuyên ngành mạng
                    semantic_words = [
                        "routing", "link-state", "distance-vector",
                        "định tuyến", "giao thức", "topology",
                        "ospf", "bgp", "rip", "eigrp",
                        "subnet", "vlan", "switching", "forwarding",
                        "bandwidth", "latency", "congestion",
                        "tcp", "udp", "ip", "mac", "arp",
                        "firewall", "nat", "dns", "dhcp"
                    ]
                    semantic_bonus = sum(
                        1 for s in semantic_words
                        if s in clean_content
                    )

                    rerank_score = (
                        distinct_match * 6
                        + exact_match * 4
                        + partial_match * 1
                        + density
                        + source_bonus
                        + semantic_bonus
                    )

                    d["rerank_score"] = rerank_score

                final_results.sort(
                    key=lambda x: x["rerank_score"],
                    reverse=True
                )

            # =========================
            # LOẠI BỎ TRANG TRÙNG LẶP
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

                if len(filtered_docs) == 6:
                    break

            top_docs = filtered_docs

            # =========================
            # KIỂM TRA KẾT QUẢ TRUY XUẤT
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
        # XÂY DỰNG CONTEXT
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
        # SYSTEM PROMPT
        # =========================
        system_prompt = f"""
Bạn là trợ lý AI chuyên ngành Mạng Máy Tính.

NHIỆM VỤ:
- Bạn phải đọc NGỮ CẢNH được cung cấp. Nếu ngữ cảnh chứa thông tin trả lời được câu hỏi, hãy tổng hợp lại.

QUY TẮC TRẢ LỜI:
1. TRƯỜNG HỢP CÓ THÔNG TIN TRONG NGỮ CẢNH: 
- Luôn trả lời bằng tiếng Việt
- Trả lời chi tiết, đầy đủ và cặn kẽ, tổng hợp thông tin từ nhiều nguồn được cung cấp. Tránh trả lời quá ngắn gọn.
- TUYỆT ĐỐI CHỈ sử dụng thông tin có trong NGỮ CẢNH được cung cấp.
- KHÔNG tìm kiếm dữ liệu bên ngoài, KHÔNG tự suy diễn, KHÔNG tự bịa ra thông tin (hallucination).
- Đảm bảo tính chính xác tuyệt đối theo tài liệu, tránh mọi sai lệch về thuật ngữ kỹ thuật.
- Nếu người dùng yêu cầu "so sánh", hãy trình bày theo từng tiêu chí rõ ràng (ví dụ: dùng bảng hoặc danh sách liệt kê).
- Trình bày mạch lạc, dễ hiểu, kết hợp các ý chính từ tài liệu một cách toàn diện thay vì chỉ sao chép nguyên văn.
- Giải thích theo phong cách giảng dạy cho sinh viên CNTT.
- Khi trả lời, ưu tiên mở đầu bằng: "Theo tài liệu ..."


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
        # SINH CÂU TRẢ LỜI (Streaming)
        # =========================
        try:

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3
            )

            thinking_placeholder.empty()

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
            # HIỂN THỊ TRÍCH DẪN NGUỒN
            # =========================
            if "Rất tiếc" in full_response:
                with st.expander("Xem nguồn trích dẫn từ giáo trình"):
                    st.info("Hệ thống đã truy xuất tài liệu nhưng không tìm thấy đoạn văn bản nào có ngữ nghĩa phù hợp với câu hỏi của bạn.")
                    if debug_mode:
                        st.divider()
                        st.markdown("**[DEBUG MODE] Các ngữ cảnh thô bị LLM từ chối:**")
                        for idx, d in enumerate(top_docs):
                            src  = d['metadata'].get('source', 'Không rõ')
                            page = d['metadata'].get('page', '?')
                            preview = d['content'][:400]
                            st.warning(
                                f"**📖 [{idx+1}] {src}** — Trang {page}\n\n"
                                f"{preview}\u2026"
                            )
            else:
                with st.expander("Xem nguồn trích dẫn từ giáo trình"):
                    for idx, d in enumerate(top_docs[:3]):
                        src  = d['metadata'].get('source', 'Không rõ')
                        page = d['metadata'].get('page', '?')
                        preview = d['content'][:400]
                        st.success(
                            f"**📖 [{idx+1}] {src}** — Trang {page}\n\n"
                            f"{preview}\u2026"
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "citations": top_docs if "Rất tiếc" in full_response else top_docs[:3]
            })

        except Exception as e:

            thinking_placeholder.empty()
            st.error(f"Lỗi AI: {e}")