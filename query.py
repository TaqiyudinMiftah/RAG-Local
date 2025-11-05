import chromadb
import sys
import time
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
try:
    from ollama._types import ResponseError as OllamaResponseError
except Exception:

    OllamaResponseError = Exception
import traceback

Settings.llm = Ollama(model="qwen3:8b") 
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

print("Menghubungkan ke database vektor yang ada...")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_collection("rag_lokal_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

query_engine = index.as_query_engine(streaming=True)

print("Database terhubung. Silakan masukkan pertanyaan Anda.")
print("Ketik 'exit' atau 'quit' untuk keluar.")
print("-" * 50)

while True:
    query = input("Pertanyaan: ")
    if query.lower() in ['exit', 'quit']:
        print("Terima kasih, sampai jumpa!")
        break
    

    print("Melakukan retrieval...")
    retrieval_start_time = time.perf_counter()
    

    retrieved_nodes = query_engine.retrieve(query)
    
    retrieval_end_time = time.perf_counter()
    retrieval_duration = retrieval_end_time - retrieval_start_time

    print(f"(Waktu Retrieval: {retrieval_duration:.2f} detik)")


    if retrieved_nodes:
        print("Sumber yang diambil:")
        for node in retrieved_nodes:
            print(f"- Skor: {node.score:.2f} | Teks: {node.text[:50]}...")
    else:
        print("Tidak ada sumber relevan yang ditemukan.")


    print("Menghasilkan jawaban (Generation)...")
    gen_start_time = time.perf_counter()


    response_stream = query_engine.synthesize(query, retrieved_nodes)
    

    try:
        response_stream.print_response_stream()
    except OllamaResponseError as e:
    
        err_text = str(e)
        print("\nTerjadi kesalahan saat menghasilkan jawaban dari Ollama.")
        if 'memory layout cannot be allocated' in err_text.lower():
            print("Penyebab kemungkinan: model terlalu besar untuk memori yang tersedia pada mesin Anda.")
        
        else:
            print("Pesan error Ollama:\n", err_text)
            traceback.print_exc()
    except Exception as e:
        print("\nTerjadi kesalahan tak terduga saat menampilkan respons:", e)
        traceback.print_exc()
    
    gen_end_time = time.perf_counter()
    gen_duration = gen_end_time - gen_start_time
    

    print(f"\n(Waktu Generation: {gen_duration:.2f} detik)")
    print(f"(Total waktu: {retrieval_duration + gen_duration:.2f} detik)")
    
    print("\n" + "-" * 50)