import streamlit as st
import os
import re
import time

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

# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.title("⚙️ Cấu hình hệ thống")

    api_key = st.text_input(
        "Nhập Gemini API Key:",
        type="password"
    )

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

    st.divider()

    st.info(
        "Hệ thống RAG sử dụng dữ liệu từ giáo trình "
        "Thiết kế mạng, CCNA và Computer Networking."
    )

# =========================
# MAIN UI
# =========================
st.title("🤖 Trợ lý ảo Mạng Máy Tính (RAG)")
st.caption("Hỗ trợ giải đáp kiến thức dựa trên giáo trình chuyên ngành")

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
# INIT DATA
# =========================
try:

    vector_db, documents, metadatas = load_resources()

    tokenized_corpus = [
        clean_text(doc).split()
        for doc in documents
    ]

    bm25 = BM25Okapi(tokenized_corpus)

except Exception as e:

    st.error(f"Lỗi tải dữ liệu: {e}")

    st.stop()

# =========================
# CHAT HISTORY
# =========================
if "messages" not in st.session_state:

    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =========================
# USER INPUT
# =========================
if prompt := st.chat_input("Hỏi em bất cứ điều gì về mạng..."):

    if not api_key:

        st.warning("⚠️ Vui lòng nhập Gemini API Key.")

        st.stop()

    # =========================
    # USER MESSAGE
    # =========================
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):

        st.markdown(prompt)

    # =========================
    # ASSISTANT
    # =========================
    with st.chat_message("assistant"):

        with st.status(
            "🔍 Đang lục tìm tài liệu...",
            expanded=False
        ) as status:

            time.sleep(1)

            # =========================
            # 1. VECTOR SEARCH
            # =========================
            vector_results = vector_db.similarity_search(
                prompt,
                k=6
            )

            # =========================
            # 2. BM25 SEARCH
            # =========================
            tokenized_query = clean_text(prompt).split()

            scores = bm25.get_scores(tokenized_query)

            top_bm25_idx = sorted(
                range(len(scores)),
                key=lambda i: scores[i],
                reverse=True
            )[:6]

            # =========================
            # 3. COMBINE RESULTS
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
            # 4. REMOVE DUPLICATE CONTENT
            # =========================
            seen_content = set()

            final_results = []

            for d in combined:

                if d["content"] not in seen_content:

                    final_results.append(d)

                    seen_content.add(d["content"])

            # =========================
            # 5. RERANKING
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

            if tu_khoa:

                technical_keywords = [
                    w for w in tu_khoa
                    if "/" in w or w.isupper() or len(w) > 4
                ]

                if technical_keywords:

                    tu_khoa_chinh = max(
                        technical_keywords,
                        key=len
                    )

                else:

                    tu_khoa_chinh = max(
                        tu_khoa,
                        key=len
                    )

                final_results.sort(
                    key=lambda x: (
                        tu_khoa_chinh in clean_text(x["content"]),
                        sum(
                            clean_text(x["content"]).count(w)
                            for w in tu_khoa
                        )
                    ),
                    reverse=True
                )

            # =========================
            # 6. REMOVE DUPLICATE PAGES
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

                if len(filtered_docs) == 3:
                    break

            top_docs = filtered_docs

            # =========================
            # 7. REFUSAL CHECK
            # =========================
            if (
                not top_docs
                or len(top_docs[0]["content"]) < 50
            ):

                st.warning(
                    "⚠️ Không tìm thấy thông tin phù hợp trong tài liệu."
                )

                st.stop()

            status.update(
                label="Đang nhờ AI đọc và phân tích độ phù hợp...",
            )

       
        # =========================
        # BUILD CONTEXT
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
- Tùyệt đối không tự suy diễn kiến thức bên ngoài internet.

QUY TẮC TRẢ LỜI:
1. TRƯỜNG HỢP CÓ THÔNG TIN TRONG NGỮ CẢNH: 
- Luôn trả lời bằng tiếng Việt
- Trả lời ngắn gọn, chính xác, dễ hiểu
- Ưu tiên định nghĩa khái niệm trước
- Giải thích theo phong cách giảng dạy cho sinh viên CNTT
- Nếu xuất hiện thuật ngữ viết tắt, hãy giải thích đầy đủ tên thuật ngữ
- Khi trả lời, hãy mở đầu bằng: "Theo tài liệu ..."
- Chủ động nhắc tên tài liệu tham khảo nếu có
- Cuối câu trả lời phải có mục:"Nguồn tham khảo: [Tên tài liệu], Trang [số trang]" để người dùng dễ dàng tra cứu
- Không suy đoán
- Không bịa thông tin

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
        # GENERATION
        # =========================
        try:

            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0.3
            )

            response = llm.invoke(system_prompt)

            full_response = response.content
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
            st.markdown(full_response)

# =========================
            # CITATION & DISPLAY LOGIC
            # =========================
            if "Rất tiếc" in full_response:
                # Nếu AI từ chối, chỉ hiện câu từ chối, KHÔNG hiện nguồn trích dẫn
                with st.expander("📚 Xem nguồn trích dẫn từ giáo trình"):
                    st.info("Hệ thống đã truy xuất tài liệu nhưng không tìm thấy đoạn văn bản nào có ngữ nghĩa phù hợp với câu hỏi của bạn.")
            else:
                # Nếu AI trả lời bình thường, hiện danh sách trích dẫn
                 # =========================
        # DEBUG PANEL
        # =========================
                with st.expander("🛠 Debug Retrieval"):

                    for i, d in enumerate(top_docs):

                        st.write(f"TOP {i+1}")

                        st.write(
                            f"Nguồn: {d['metadata']}"
                        )

                        st.caption(
                            d["content"][:300]
                        )

                with st.expander("📚 Xem nguồn trích dẫn từ giáo trình"):
                    for d in top_docs:
                        st.write(
                            f"- **{d['metadata'].get('source')}** "
                            f"(Trang {d['metadata'].get('page')})"
                        )
                        st.caption(
                            f"Nội dung: {d['content'][:150]}..."
                        )

            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response
            })

        except Exception as e:

            st.error(f"Lỗi AI: {e}")