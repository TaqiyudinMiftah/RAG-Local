import chromadb
import time  
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

print("Memulai proses indeksasi...")

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

print("Menghubungkan ke ChromaDB di ./chroma_db")
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("rag_lokal_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("Memuat dokumen dari folder './data'...")
load_start_time = time.perf_counter()  

documents = SimpleDirectoryReader("./data").load_data()

load_end_time = time.perf_counter()  
load_duration = load_end_time - load_start_time
print(f"Berhasil memuat {len(documents)} dokumen dalam {load_duration:.2f} detik.")

print("Membuat indeks dan menyimpan embedding (proses ini mungkin memakan waktu)...")
index_start_time = time.perf_counter()  

index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

index_end_time = time.perf_counter() 
index_duration = index_end_time - index_start_time

print("-" * 50)
print(f"Indeksasi selesai dalam {index_duration:.2f} detik.")  # <-- 6. Tampilkan durasi
print(f"Total waktu (termasuk memuat): {load_duration + index_duration:.2f} detik.")
print("Indeksasi selesai dan disimpan di ./chroma_db")
print("Anda sekarang dapat menjalankan query.py untuk bertanya.")