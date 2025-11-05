import chromadb
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core import Settings

print("Memulai proses indeksasi...")

# 1. Tentukan model embedding yang akan digunakan
# Pastikan Anda sudah 'ollama pull nomic-embed-text'
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# 2. Inisialisasi ChromaDB (Persistent)
# Ini akan menyimpan data di disk dalam folder './chroma_db'
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("rag_lokal_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. Muat dokumen dari folder 'data'
# Ini akan otomatis membaca .pdf, .txt, .md, .docx, dll.
documents = SimpleDirectoryReader("./data").load_data()
print(f"Berhasil memuat {len(documents)} dokumen.")

# 4. Buat Indeks
# Proses ini akan memecah dokumen, membuat embedding, dan menyimpannya di ChromaDB
print("Membuat indeks dan menyimpan embedding...")
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

print("-" * 50)
print("Indeksasi selesai dan disimpan di ./chroma_db")
print("Anda sekarang dapat menjalankan query.py untuk bertanya.")