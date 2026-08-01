# ⚡ Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites
- Python 3.10+
- Ollama installed

## Commands

### 1️⃣ Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Setup Environment
```bash
cp .env.example .env
```

### 4️⃣ Start Ollama (in a new terminal)
```bash
# First time only:
ollama pull llama3:8b

# Then start server:
ollama serve
```

### 5️⃣ Add Your PDFs
```bash
# Copy your PDF files to:
# college_data/
```

### 6️⃣ Process Documents
```bash
python ingest.py
```

### 7️⃣ Launch App
```bash
streamlit run app.py
```

✅ **Done!** Open http://localhost:8501 in your browser.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ollama: command not found` | [Install Ollama](https://ollama.ai/) |
| `No module named streamlit` | Run `pip install -r requirements.txt` |
| Ollama connection error | Run `ollama serve` in another terminal |
| Model not found | Run `ollama pull llama3:8b` |

## Next Steps
- Read [README.md](README.md) for full documentation
- Customize settings in the Streamlit sidebar
- Try different Ollama models: `ollama pull mistral`, `ollama pull neural-chat`

**Happy learning! 📚**
