import streamlit as st
import chromadb
from llama_index.core import VectorStoreIndex, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.memory import ChatMemoryBuffer

# --- KONFIGURASI APLIKASI ---
st.set_page_config(
    page_title="Chat RAG Lokal",
    page_icon="💬",
    layout="centered",
)

# --- FUNGSI CACHE UNTUK MEMUAT MODEL & INDEKS ---
@st.cache_resource(show_spinner="Menghubungkan ke database dan memuat model...")
def load_rag_pipeline():
    """
    Memuat pipeline RAG (indeks, LLM, embedding) dari database.
    """
    try:
        # 1. Tentukan model (LLM & Embedding)
        Settings.llm = Ollama(model="qwen3:8b", request_timeout=300.0)
        Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text") # NAMA MODEL ANDA DI SINI

        # 2. Hubungkan ke ChromaDB yang sudah ada
        db = chromadb.PersistentClient(path="./chroma_db")
        chroma_collection = db.get_collection("rag_lokal_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # 3. Muat indeks dari vector store
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        return index
    
    except Exception as e:
        st.error(f"Gagal memuat pipeline RAG: {e}\n\n"
                 f"Pastikan Anda sudah menjalankan 'index.py' dan folder './chroma_db' ada.")
        st.stop()

# --- INISIALISASI ---
index = load_rag_pipeline()

# Inisialisasi memori obrolan (disimpan dalam session state)
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

# Buat chat engine, hubungkan dengan memori
chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=st.session_state.memory,
    streaming=True,
    verbose=True 
)

# --- ANTARMUKA (UI) STREAMLIT ---
st.title("💬 Chat RAG Lokal Anda")
# --- PERBAIKAN DI BAWAH INI ---
# Menggunakan _llm (private attribute) sesuai pesan error
st.caption(f"Didukung oleh Ollama `{chat_engine._llm.model}` dan dokumen lokal Anda.")
# --- PERBAIKAN DI ATAS ---

# Inisialisasi riwayat obrolan di session state
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant", 
        "content": "Halo! Saya adalah asisten RAG lokal Anda. Silakan tanya apa saja tentang dokumen Anda."
    }]

# Tampilkan riwayat obrolan
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # Tampilkan sumber jika ada
        if "sources" in msg and msg["sources"]:
            with st.expander("Lihat Sumber Jawaban"):
                for i, source in enumerate(msg["sources"]):
                    st.info(f"Sumber {i+1} (Skor: {source.score:.2f})\n\n"
                            f"**File:** `{source.metadata.get('file_name', 'N/A')}`\n\n"
                            f"**Teks:**\n{source.text[:250]}...")

# --- LOGIKA INPUT DAN RESPON OBROLAN ---
if prompt := st.chat_input("Apa pertanyaan Anda?"):
    # 1. Tambahkan dan tampilkan pesan pengguna
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Hasilkan dan tampilkan respons asisten (streaming)
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        
        # Panggil chat_engine secara streaming
        streaming_response = chat_engine.stream_chat(prompt)
        
        # Iterasi melalui token jawaban
        for chunk in streaming_response.response_gen:
            full_response += chunk
            response_placeholder.markdown(full_response + "▌") # Tampilkan kursor
        
        response_placeholder.markdown(full_response) # Jawaban final

        # 3. Ambil dan tampilkan sumber
        sources = streaming_response.source_nodes
        assistant_message = {"role": "assistant", "content": full_response}
        
        if sources:
            with st.expander("Lihat Sumber Jawaban"):
                for i, source in enumerate(sources):
                    st.info(f"Sumber {i+1} (Skor: {source.score:.2f})\n\n"
                            f"**File:** `{source.metadata.get('file_name', 'N/A')}`\n\n"
                            f"**Teks:**\n{source.text[:250]}...")
            # Simpan sumber ke session state
            assistant_message["sources"] = sources

    # 4. Tambahkan respons asisten (lengkap dengan sumber) ke riwayat
    st.session_state.messages.append(assistant_message)