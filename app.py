# app.py
import streamlit as st
import chromadb
import time
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler

# =========================
# KONFIGURASI APLIKASI
# =========================
st.set_page_config(
    page_title="Chat RAG Lokal",
    page_icon="💬",
    layout="centered",
)

# =========================
# GLOBAL SETTINGS (LLM & EMBEDDING)
# =========================
Settings.llm = Ollama(model="qwen3:8b", request_timeout=300.0)
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# =========================
# UTIL: ambil debug log lintas-versi
# =========================
def _extract_debug_log(handler) -> str:
    try:
        if hasattr(handler, "get_event_logs_str") and callable(handler.get_event_logs_str):
            return handler.get_event_logs_str()
        if hasattr(handler, "get_logs_str") and callable(handler.get_logs_str):
            return handler.get_logs_str()
        if hasattr(handler, "get_event_logs") and callable(handler.get_event_logs):
            logs = handler.get_event_logs()
            return "\n".join(map(str, logs)) if logs else "(kosong)"
        if hasattr(handler, "get_events") and callable(handler.get_events):
            events = handler.get_events()
            return "\n".join(map(str, events)) if events else "(kosong)"
        if hasattr(handler, "event_pairs"):
            pairs = getattr(handler, "event_pairs", [])
            lines = []
            for pair in pairs or []:
                try:
                    start = pair[0]
                    end = pair[1] if len(pair) > 1 else None
                    s_type = getattr(start, "event_type", "start")
                    s_payload = getattr(start, "payload", {})
                    lines.append(f"> {s_type}: {s_payload}")
                    if end is not None:
                        e_type = getattr(end, "event_type", "end")
                        e_payload = getattr(end, "payload", {})
                        lines.append(f"  └─ {e_type}: {e_payload}")
                except Exception:
                    lines.append(str(pair))
            return "\n".join(lines) if lines else "(kosong)"
    except Exception as e:
        return f"(gagal mengambil log: {e})"
    return "(debug log tidak tersedia pada versi LlamaIndex ini)"

# =========================
# MEMUAT INDEKS DARI CHROMADB (CACHE)
# =========================
@st.cache_resource(show_spinner="Menghubungkan ke database vektor...")
def load_index_from_db():
    try:
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_collection("rag_lokal_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        return index
    except Exception as e:
        st.error(
            "Gagal memuat pipeline RAG:\n\n"
            f"{e}\n\n"
            "Pastikan Anda sudah menjalankan skrip indexing (mis. index.py) dan folder './chroma_db' beserta "
            "koleksi 'rag_lokal_collection' tersedia."
        )
        st.stop()

# =========================
# INISIALISASI
# =========================
index = load_index_from_db()

# Memori obrolan (bertahan di session)
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Riwayat pesan untuk UI chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Halo! Saya asisten RAG lokal Anda. Silakan tanya apa saja tentang dokumen Anda."
    }]

# =========================
# RENDER BANTUAN
# =========================
def _render_sources(sources):
    if not sources:
        return
    with st.expander("Lihat Sumber Jawaban (Konteks)"):
        for i, src in enumerate(sources):
            score = getattr(src, "score", None)
            skor_str = f"{score:.2f}" if isinstance(score, (int, float)) else "N/A"

            # Ambil isi teks lintas-versi
            text = None
            try:
                text = getattr(src, "text", None)
                if text is None and hasattr(src, "get_content"):
                    text = src.get_content()
                if text is None and hasattr(src, "node") and hasattr(src.node, "get_content"):
                    text = src.node.get_content()
            except Exception:
                text = None

            # Ambil metadata lintas-versi
            metadata = {}
            try:
                metadata = getattr(src, "metadata", None) or getattr(getattr(src, "node", None), "metadata", {}) or {}
            except Exception:
                metadata = {}

            file_name = metadata.get("file_name", "N/A")
            preview = (text or "").strip().replace("\n", " ")
            if len(preview) > 250:
                preview = preview[:250] + "..."

            st.info(
                f"Sumber {i+1} (Skor: {skor_str})\n\n"
                f"**File:** `{file_name}`\n\n"
                f"**Teks:**\n{preview or '(tidak ada preview)'}"
            )

def _render_safe_think(stage, stage_title, lines):
    """Render list 'reasoning trace' yang aman dibagikan (tanpa context manager)."""
    stage.markdown(f"**{stage_title}**")
    for line in lines:
        stage.write(f"- {line}")


# =========================
# ANTARMUKA
# =========================
st.title("💬 Chat RAG Lokal Anda")
st.caption(f"Didukung oleh Ollama `{Settings.llm.model}` dan dokumen lokal Anda.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "safe_trace" in msg and msg["safe_trace"]:
            with st.expander("Reasoning (Safe Trace)"):
                _render_safe_think(st, "Langkah", msg["safe_trace"])
        if "sources" in msg and msg["sources"]:
            _render_sources(msg["sources"])
        if "debug_log" in msg and msg["debug_log"]:
            with st.expander("Verbose Debug Log"):
                log_text = msg["debug_log"].replace(">", "    >")
                st.code(log_text, language="text")

# =========================
# LOGIKA INPUT & RESPON
# =========================
prompt = st.chat_input("Apa pertanyaan Anda?")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ========== SAFE THINK (ditampilkan lebih dulu) ==========
    # Handler & callback untuk jejak proses
    local_debug_handler = LlamaDebugHandler(print_trace_on_end=False)
    local_callback_manager = CallbackManager([local_debug_handler])
    old_cb = getattr(Settings, "callback_manager", None)
    Settings.callback_manager = local_callback_manager

    # 1) Mulai stopwatch
    t0 = time.time()

    # 2) Lakukan RETRIEVAL TERLEBIH DULU (sebagai "think" yang aman)
    #    Kita tampilkan file, skor, dan preview SEBELUM jawaban di-stream.
    safe_trace_lines = []
    safe_block = st.container()
    with st.chat_message("assistant"):
        trace_placeholder = st.empty()  # tempat untuk menaruh panel "Reasoning (Safe Trace)"
        with st.spinner("Menganalisis pertanyaan & mengambil konteks..."):
            try:
                retriever = index.as_retriever(similarity_top_k=5)
                retrieved_nodes = retriever.retrieve(prompt)
            except Exception as e:
                retrieved_nodes = []
                safe_trace_lines.append(f"Gagal melakukan retrieval: {e}")

        # Rangkai panel safe trace (tanpa membuka isi prompt/response internal model)
        elapsed_retrieve = time.time() - t0
        if retrieved_nodes:
            safe_trace_lines.append(f"Menemukan {len(retrieved_nodes)} konteks teratas (top-k).")
            for i, n in enumerate(retrieved_nodes, 1):
                score = getattr(n, "score", None)
                skor_str = f"{score:.3f}" if isinstance(score, (int, float)) else "N/A"
                meta = getattr(n, "metadata", None) or getattr(getattr(n, "node", None), "metadata", {}) or {}
                file_name = meta.get("file_name", "N/A")
                safe_trace_lines.append(f"Konteks {i}: file=`{file_name}` | skor={skor_str}")
        else:
            safe_trace_lines.append("Tidak ada konteks yang ditemukan atau retrieval gagal.")

        safe_trace_lines.append(f"Waktu retrieval: {elapsed_retrieve:.2f}s")
        # Render “Reasoning (Safe Trace)” SEBELUM streaming jawaban
        with trace_placeholder.container():
            st.markdown("### 🧭 Reasoning (Safe Trace)")
            _render_safe_think(st, "Langkah", safe_trace_lines)

    # ========== STREAMING JAWABAN ==========
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        # Buat chat engine TANPA mengirim callback_manager (hindari argumen ganda)
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=st.session_state.memory,
            streaming=True,
        )

        try:
            t1 = time.time()
            streaming_response = chat_engine.stream_chat(prompt)

            for chunk in streaming_response.response_gen:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)

            # Sumber & log lengkap
            sources = getattr(streaming_response, "source_nodes", [])
            debug_log = _extract_debug_log(local_debug_handler)

            # Tambahkan “Reasoning (Safe Trace)” ke message history agar bisa dilihat ulang
            assistant_message = {
                "role": "assistant",
                "content": full_response,
                "sources": sources if sources else [],
                "debug_log": debug_log,
                "safe_trace": safe_trace_lines + [f"Waktu generasi jawaban: {time.time()-t1:.2f}s"]
            }
        except Exception as e:
            st.error(f"Terjadi kesalahan saat menghasilkan jawaban: {e}")
            assistant_message = {
                "role": "assistant",
                "content": "Maaf, terjadi kesalahan. Coba lagi.",
                "safe_trace": safe_trace_lines
            }
        finally:
            Settings.callback_manager = old_cb  # pulihkan

    st.session_state.messages.append(assistant_message)

# =========================
# Catatan:
# - “Reasoning (Safe Trace)” menampilkan langkah & hasil retrieval (file, skor, waktu) SEBELUM jawaban.
# - Tidak menampilkan chain-of-thought/prompt internal model, hanya jejak proses yang aman.
# - Pastikan Ollama dan model tersedia, serta ./chroma_db sudah berisi koleksi.
# =========================
