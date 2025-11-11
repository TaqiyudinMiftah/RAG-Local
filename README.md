# 🧠 RAG-Local

A fully local implementation of **Retrieval-Augmented Generation (RAG)** using **LlamaIndex**, **ChromaDB**, and **Streamlit**.

This project demonstrates how to build a **privacy-preserving, offline AI system** capable of understanding and answering questions from your own documents — without depending on cloud APIs or internet connectivity.

---

## 📘 Overview

**RAG-Local** combines **retrieval** and **generation** to enhance local large language models with your own custom knowledge base.
It follows the core stages of the RAG pipeline:

1. **Document Loading** — Import and preprocess your local text or PDF files.
2. **Vectorization** — Generate embeddings using **Ollama embeddings** and store them in **ChromaDB**.
3. **Retrieval** — Search for semantically relevant chunks based on a user query.
4. **Generation** — Produce accurate, context-aware answers using a **local Ollama LLM**.

---

## ⚙️ Project Structure

```
RAG Local/
├── app.py              # Streamlit web interface for user interaction
├── index.py            # Handles document loading and vector database creation
├── query.py            # Performs retrieval and generation using LlamaIndex
├── requirements.txt    # Python dependencies
├── data/               # Directory for source documents
├── chroma_db/          # Local vector database storage
└── README.md
```

---

## 🧩 Features

✅ 100% local — no external API keys required
✅ Uses **LlamaIndex** for flexible RAG pipelines
✅ Uses **Ollama** for both LLM and embedding models
✅ Stores vectors locally in **ChromaDB**
✅ Includes a **Streamlit** UI for simple interaction
✅ Modular and easy to extend

---

## 💻 Installation

1. **Clone this repository**

   ```bash
   git clone https://github.com/<your-username>/RAG-Local.git
   cd RAG-Local
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate      # On Linux/Mac
   venv\Scripts\activate         # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ Installing Ollama and Models

To run this project, you need **Ollama** installed on your machine. Ollama provides local execution for LLMs and embedding models.

### 🧩 Step 1 — Install Ollama

Follow the installation instructions from the official site:
👉 [https://ollama.ai/download](https://ollama.ai/download)

After installation, verify it works:

```bash
ollama --version
```

### 🧠 Step 2 — Pull Required Models

This project uses two models:

* **`qwen3:8b`** — Main LLM for text generation and reasoning.
* **`nomic-embed-text`** — Embedding model for converting text into vector representations.

Download both models locally using the commands below:

```bash
ollama pull qwen3:8b
ollama pull nomic-embed-text
```

You can verify the models are available by running:

```bash
ollama list
```

This will display a list of installed models, including `qwen3:8b` and `nomic-embed-text`.

### 🧩 Optional — Create a Custom Modelfile

If you want to configure your own model parameters or prompt templates, you can create a `Modelfile` like this:

```Dockerfile
FROM qwen3:8b
PARAMETER temperature 0.7
PARAMETER top_p 0.9
```

Then build it with:

```bash
ollama create my-qwen3 -f Modelfile
```

You can use your custom model name (e.g., `my-qwen3`) in the code.

---

## 🚀 Usage

### 🏗️ Step 1 — Build the Knowledge Base

Run the following command to load and embed your documents:

```bash
python index.py
```

This will process files in the `data/` folder and create a vector database in `chroma_db/`.

---

### 🔍 Step 2 — Ask Questions

Once the database is ready, you can query it:

```bash
python query.py
```

Type your question, and the system will retrieve the most relevant information and generate a context-aware answer using the **Ollama LLM**.

---

### 🧠 Step 3 — Run the Streamlit App

You can launch the web-based interface with:

```bash
streamlit run app.py
```

This will open a local web interface for interactive querying.

---

## 🧰 Tech Stack

* **Python 3.10+**
* **LlamaIndex** — for managing documents, retrieval, and query pipelines
* **llama-index-llms-ollama** — for local LLM integration
* **llama-index-embeddings-ollama** — for local embeddings
* **llama-index-vector-stores-chroma** — for ChromaDB integration
* **ChromaDB** — local vector store
* **Streamlit** — lightweight UI framework
* **Ollama** — local LLM and embedding model runner

---

## 🧠 Models Used

This project uses **Ollama** to run both the LLM and embedding models locally:

* **`qwen3:8b`** — A powerful and efficient open-source large language model from Alibaba’s Qwen3 family, used for **text generation and reasoning**. It provides excellent performance for retrieval-augmented generation tasks while remaining efficient enough for local execution.

* **`nomic-embed-text`** — An embedding model from Nomic AI used to **convert text into high-dimensional vector representations**, enabling efficient semantic search and retrieval through ChromaDB.

These models are fully compatible with **Ollama**, allowing for smooth local deployment without external API dependencies.

---

## 🧩 Example Use Case

> You can place your research papers, notes, or project documents inside the `data/` folder.
> Then, simply ask questions like:
>
> “What are the main topics discussed in the file data.txt?”
>
> and the system will retrieve relevant content and summarize it locally using **qwen3:8b** and **nomic-embed-text**.

---

## 🏁 Goal

This repository was created as part of my learning journey to understand and implement **Retrieval-Augmented Generation (RAG)** locally using **LlamaIndex** and **Ollama**.
It serves as a foundation for building fully local, private, and efficient AI knowledge systems.

---

## 👨‍💻 Author

**Taqiyudin Miftah Adn**
📚 Computer Engineering student passionate about **Artificial Intelligence** and **Information Retrieval Systems**.
