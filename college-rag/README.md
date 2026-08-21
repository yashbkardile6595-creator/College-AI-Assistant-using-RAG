# 📚 College RAG Assistant

A local, privacy-first Retrieval-Augmented Generation (RAG) application for college documents. Built with Python, Streamlit, LangChain, ChromaDB, and Ollama—**zero paid APIs**.

## ✨ Features

- 📄 **Multiple PDF uploads** — Process multiple college documents at once
- 🔍 **Semantic search** — Find relevant information using ChromaDB vector embeddings
- 🤖 **Local LLM** — Uses Ollama with Llama 3:8B (runs entirely on your machine)
- 💬 **Chat history** — Persistent conversation history across sessions
- 📚 **Source references** — See which documents your answers come from
- ⚡ **Modern UI** — Clean, intuitive Streamlit interface
- 🆓 **No APIs required** — Everything runs locally, no external dependencies

## 🏗️ Tech Stack

- **Frontend**: Streamlit
- **LLM Framework**: LangChain
- **Vector Database**: ChromaDB
- **Embeddings**: Hugging Face (`sentence-transformers/all-MiniLM-L6-v2`)
- **LLM**: Ollama (Llama 3:8B)
- **PDF Processing**: PyPDF
- **Language**: Python 3.10+

## 📋 Prerequisites

Before you begin, ensure you have:

1. **Python 3.10+** — [Download](https://www.python.org/downloads/)
2. **Ollama** — [Download](https://ollama.ai/)
3. **Git** (optional)

## 🚀 Quick Start

### Step 1: Clone or Extract the Project

```bash
# If you have the zip file, extract it
unzip college-rag.zip
cd college-rag

# Or clone from git (if applicable)
git clone <repo-url>
cd college-rag
```

### Step 2: Set Up Python Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env if needed (default values should work)
# OLLAMA_API_URL=http://localhost:11434
# OLLAMA_MODEL=llama3:8b
```

### Step 5: Download and Run Ollama

```bash
# Download Ollama from https://ollama.ai/

# After installing, pull the Llama 3:8B model in a new terminal:
ollama pull llama3:8b

# Then start the Ollama server (keep this running):
ollama serve
```

### Step 6: Add Your College Documents

```bash
# Place your PDF files in the college_data/ folder
# Example:
# college_data/
# ├── admissions_guide.pdf
# ├── course_catalog.pdf
# └── student_handbook.pdf
```

### Step 7: Ingest Documents into ChromaDB

```bash
# Process all PDFs in college_data/ folder
python ingest.py
```

Expected output:
```
✅ Found 3 PDF file(s)
🧠 Initializing embeddings...
✅ Embeddings initialized
📦 Initializing ChromaDB...
✅ ChromaDB initialized
📄 Loading and splitting documents...
✅ Loaded 156 document chunks
💾 Adding documents to database...
✅ Added 156 documents to ChromaDB
🎉 Ingestion complete!
```

### Step 8: Launch the Application

```bash
streamlit run app.py
```

Your browser will open to `http://localhost:8501` 🎉

## 📖 Usage Guide

### 1. **Initialize Database**
   - Click the "🔄 Initialize Database" button in the sidebar
   - This sets up ChromaDB connection

### 2. **Upload Documents** (if not already done)
   - Click "📤 Process & Store Documents"
   - Select PDF files and upload
   - Wait for the processing to complete

### 3. **Ask Questions**
   - Type your question in the chat input field
   - Example questions:
     - "What's the admission process?"
     - "Tell me about scholarship opportunities"
     - "What are the requirements for graduation?"

### 4. **View Responses**
   - Assistant responds with information from your documents
   - See source references showing which documents were used
   - Adjust settings in the sidebar (temperature, top-k retrieval)

### 5. **Manage Chat History**
   - Chat history is automatically saved
   - Click "🗑️ Clear Chat History" to start fresh

## ⚙️ Configuration

### Settings in Sidebar

| Setting | Range | Default | Description |
|---------|-------|---------|-------------|
| **Temperature** | 0.0 - 1.0 | 0.7 | Controls response creativity (higher = more creative) |
| **Top K Documents** | 1 - 10 | 3 | Number of source documents to retrieve per query |

### Environment Variables (.env)

```bash
# Ollama API endpoint
OLLAMA_API_URL=http://localhost:11434

# Model to use (make sure you've pulled it with: ollama pull llama3:8b)
OLLAMA_MODEL=llama3:8b
```

## 🔍 Troubleshooting

### "Ollama is not running"
```bash
# In a new terminal, start Ollama:
ollama serve
```

### "Model 'llama3:8b' not found"
```bash
# Pull the model:
ollama pull llama3:8b
```

### "No module named 'streamlit'"
```bash
# Reinstall dependencies:
pip install -r requirements.txt
```

### "Slow embeddings on first run"
- First-time embedding model download takes time (~600MB)
- Subsequent runs will be much faster
- The model is cached locally

### "High memory usage"
- Llama 3:8B requires ~8GB RAM minimum
- Close other applications
- If still issues, consider `llama2:7b` (smaller model) instead:
  ```bash
  ollama pull llama2:7b
  # Update .env: OLLAMA_MODEL=llama2:7b
  ```

## 📁 Project Structure

```
college-rag/
├── app.py                 # Main Streamlit application
├── ingest.py              # PDF ingestion pipeline
├── utils.py               # Utility functions
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── .env                   # Environment variables (create from example)
├── README.md              # This file
├── chat_history.json      # Auto-generated chat history
├── college_data/          # Place your PDF files here
│   └── sample.pdf
└── chroma_db/             # Auto-generated vector database
    └── (ChromaDB files)
```

## 🎯 Advanced Usage

### Using Different Models

```bash
# Alternative models:
ollama pull llama2:7b      # Smaller, faster
ollama pull mistral        # More specialized
ollama pull neural-chat    # Optimized for chat
```

Then update `.env`:
```bash
OLLAMA_MODEL=llama2:7b
```

### Batch Processing Multiple Collections

Edit `utils.py` to use different collection names:
```python
# For different document types:
add_documents_to_chromadb(vector_db, docs, embeddings, collection_name="admissions")
add_documents_to_chromadb(vector_db, docs, embeddings, collection_name="academics")
```

### Custom Document Splitting

Adjust in `utils.py`:
```python
CHUNK_SIZE = 1000        # Increase for longer context
CHUNK_OVERLAP = 200      # Increase for better continuity
```

## 📊 Performance Tips

- **First run**: Embeddings model download (~600MB), takes 2-3 minutes
- **Subsequent runs**: Fast, fully cached
- **Response time**: Typically 5-15 seconds depending on model and document size
- **Memory**: ~8GB RAM required for Llama 3:8B

## 🛡️ Privacy & Security

✅ **100% Local** — All data stays on your machine
✅ **No Internet Required** — Works completely offline
✅ **No Data Collection** — No telemetry or tracking
✅ **No API Keys** — No external service dependencies

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📞 Support

If you encounter issues:
1. Check the **Troubleshooting** section
2. Verify all prerequisites are installed
3. Ensure Ollama is running (`ollama serve`)
4. Check logs in the terminal

## 🚀 Next Steps

- [ ] Upload your college documents to `college_data/`
- [ ] Run `python ingest.py` to process them
- [ ] Launch with `streamlit run app.py`
- [ ] Ask your first question!


