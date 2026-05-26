import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import shutil
import fitz

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")

DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# =========================
# BUILD DATABASE
# =========================
def build_multi_db():

    print("\n==============================")
    print("BUILD VECTOR DATABASE")
    print("==============================\n")

    documents = []

    pdf_stats = []

    # =========================
    # READ PDF FILES
    # =========================
    for filename in os.listdir(DATA_DIR):

        if not filename.endswith(".pdf"):
            continue

        filepath = os.path.join(DATA_DIR, filename)

        print("--------------------------------------------------")
        print(f"[READ] FILE: {filename}")

        try:

            doc = fitz.open(filepath)

            total_pages = len(doc)

            valid_pages = 0

            total_text_length = 0

            file_documents = []

            # =========================
            # READ EACH PAGE
            # =========================
            for page_num in range(total_pages):

                page = doc.load_page(page_num)

                text = page.get_text().strip()

                # Bỏ qua trang rỗng
                if len(text) > 50:

                    valid_pages += 1

                    total_text_length += len(text)

                    file_documents.append(
                        Document(
                            page_content=f"TÀI LIỆU: {filename}\n\n{text}",
                            metadata={
                                "source": filename,
                                "page": page_num + 1,
                                "chunk_id": f"{filename}_{page_num+1}"
                            }
                        )
                    )

            doc.close()

            # =========================
            # TEST CHUNKING
            # =========================
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", " ", ""]
            )

            chunks = splitter.split_documents(file_documents)

            # =========================
            # QUALITY CHECK
            # =========================
            avg_text = (
                total_text_length / valid_pages
                if valid_pages > 0
                else 0
            )

            status = "[OK]"

            warning = ""

            if valid_pages == 0:

                status = "[ERROR]"

                warning = "Khong extract duoc text"

            elif avg_text < 200:

                status = "[WARNING]"

                warning = "Text qua ngan -> PDF co the scan loi"

            elif len(chunks) < 5:

                status = "[WARNING]"

                warning = "Qua it chunk"

            # =========================
            # PRINT FILE REPORT
            # =========================
            print(f"Pages          : {total_pages}")

            print(f"Valid Pages    : {valid_pages}")

            print(f"Chunks         : {len(chunks)}")

            print(f"Avg Text/Page  : {avg_text:.1f}")

            print(f"Status         : {status}")

            if warning:
                print(f"Warning        : {warning}")

            # =========================
            # SAVE STATS
            # =========================
            pdf_stats.append({
                "file": filename,
                "pages": total_pages,
                "valid_pages": valid_pages,
                "chunks": len(chunks),
                "avg_text": avg_text,
                "status": status,
                "warning": warning
            })

            # ADD TO GLOBAL DOCUMENTS
            documents.extend(file_documents)

        except Exception as e:

            print(f"[ERROR] Loi file {filename}: {e}")

    # =========================
    # GLOBAL CHUNKING
    # =========================
    print("\n==============================")
    print("[CHUNKING] GLOBAL PROCESS")
    print("==============================")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    print(f"Total Chunks: {len(chunks)}")

    # =========================
    # EMBEDDING
    # =========================
    print("\n[EMBEDDING] Dang tao embeddings...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # =========================
    # RESET DATABASE
    # =========================
    if os.path.exists(DB_DIR):

        print("[RESET] Xoa database cu...")

        shutil.rmtree(DB_DIR)

    # =========================
    # SAVE CHROMADB
    # =========================
    print("[SAVE] Dang luu vao ChromaDB...")

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )

    # =========================
    # FINAL REPORT
    # =========================
    print("\n==============================")
    print("FINAL PDF REPORT")
    print("==============================\n")

    for stat in pdf_stats:

        print(
            f"{stat['status']} "
            f"{stat['file']} | "
            f"Pages={stat['pages']} | "
            f"Valid={stat['valid_pages']} | "
            f"Chunks={stat['chunks']}"
        )

    print("\n[DONE] BUILD DATABASE THANH CONG!")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    build_multi_db()