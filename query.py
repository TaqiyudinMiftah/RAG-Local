import chromadb
import sys
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
try:
    from ollama._types import ResponseError as OllamaResponseError
except Exception:
    # If the ollama package isn't available at edit-time, fall back to a generic Exception
    OllamaResponseError = Exception
import traceback

# 1. Tentukan model (LLM & Embedding)
# Pastikan model ini sudah di-pull dan Ollama berjalan
Settings.llm = Ollama(model="qwen3:8b") 
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 2. Hubungkan ke ChromaDB yang sudah ada
print("Menghubungkan ke database vektor yang ada...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_collection("rag_lokal_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# 3. Muat indeks dari vector store
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 4. Buat Query Engine
# 'streaming=True' akan memberikan jawaban kata per kata (seperti ChatGPT)
query_engine = index.as_query_engine(streaming=True)

print("Database terhubung. Silakan masukkan pertanyaan Anda.")
print("Ketik 'exit' atau 'quit' untuk keluar.")
print("-" * 50)

# 5. Loop untuk tanya jawab interaktif
while True:
    query = input("Pertanyaan: ")
    if query.lower() in ['exit', 'quit']:
        print("Terima kasih, sampai jumpa!")
        break
    
    print("Menghasilkan jawaban...")
    response_stream = query_engine.query(query)
    
    # Cetak respons secara streaming
    try:
        response_stream.print_response_stream()
    except OllamaResponseError as e:
        # Provide a friendly, actionable message for common Ollama memory allocation failures
        err_text = str(e)
        print("\nTerjadi kesalahan saat menghasilkan jawaban dari Ollama.")
        if 'memory layout cannot be allocated' in err_text.lower():
            print("Penyebab kemungkinan: model terlalu besar untuk memori yang tersedia pada mesin Anda.")
            print("Solusi yang disarankan:")
            print("  1) Gunakan model yang lebih kecil (pull model yang lebih kecil dengan 'ollama pull <model>').")
            print("  2) Tutup aplikasi lain untuk membebaskan RAM atau jalankan pada mesin dengan lebih banyak memori.")
            print("  3) Periksa model yang terpasang: jalankan 'ollama list' untuk melihat ukuran model.")
            print("  4) Jika masih gagal, pertimbangkan menjalankan Ollama pada server/Linux dengan lebih banyak memori.")
        else:
            print("Pesan error Ollama:\n", err_text)
            print("Traceback (local):")
            traceback.print_exc()
    except Exception as e:
        print("\nTerjadi kesalahan tak terduga saat menampilkan respons:", e)
        traceback.print_exc()
    print("\n" + "-" * 50)