"""
Utility functions for the College RAG Assistant
Handles embeddings, document processing, ChromaDB operations, and Ollama integration
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any
import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import chromadb
import requests

# Constants
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
CHROMA_DB_PATH = "chroma_db"
CHAT_HISTORY_FILE = "chat_history.json"


def initialize_embeddings():
    """Initialize Hugging Face embeddings"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
        )
        return embeddings
    except Exception as e:
        raise Exception(f"Failed to initialize embeddings: {str(e)}")


def initialize_chromadb():
    """Initialize ChromaDB"""
    try:
        # Create chroma_db directory if it doesn't exist
        Path(CHROMA_DB_PATH).mkdir(exist_ok=True)
        
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        
        return client
    except Exception as e:
        raise Exception(f"Failed to initialize ChromaDB: {str(e)}")


def load_and_split_documents(file_paths: List[str]) -> List[Dict[str, Any]]:
    """Load PDF documents and split them into chunks"""
    documents = []
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    
    for file_path in file_paths:
        try:
            # Load PDF
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Split pages into chunks
            for page in pages:
                splits = text_splitter.split_text(page.page_content)
                
                for i, split in enumerate(splits):
                    doc_dict = {
                        "content": split,
                        "metadata": {
                            "source": Path(file_path).name,
                            "page": page.metadata.get("page", 0) + 1,
                            "chunk": i,
                        }
                    }
                    documents.append(doc_dict)
        
        except Exception as e:
            print(f"⚠️  Error loading {file_path}: {str(e)}")
            continue
    
    return documents


def add_documents_to_chromadb(
    client: chromadb.PersistentClient,
    documents: List[Dict[str, Any]],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str = "college_documents"
) -> None:
    """Add documents to ChromaDB with embeddings"""
    
    try:
        # Get or create collection
        collection = client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Prepare data for ChromaDB
        ids = []
        documents_content = []
        metadatas = []
        embeddings_list = []
        
        for doc in documents:
            # Generate unique ID based on content
            doc_id = hashlib.md5(
                (doc["content"] + str(doc["metadata"])).encode()
            ).hexdigest()
            
            ids.append(doc_id)
            documents_content.append(doc["content"])
            metadatas.append(doc["metadata"])
            
            # Generate embedding
            embedding = embeddings.embed_query(doc["content"])
            embeddings_list.append(embedding)
        
        # Add to collection
        collection.add(
            ids=ids,
            documents=documents_content,
            metadatas=metadatas,
            embeddings=embeddings_list
        )
        
        print(f"✅ Added {len(documents)} documents to ChromaDB")
        
    except Exception as e:
        raise Exception(f"Failed to add documents to ChromaDB: {str(e)}")


def query_chromadb(
    client: chromadb.PersistentClient,
    query: str,
    embeddings: HuggingFaceEmbeddings,
    k: int = 3,
    collection_name: str = "college_documents"
) -> List[Dict[str, Any]]:
    """Query ChromaDB for relevant documents"""
    
    try:
        collection = client.get_collection(name=collection_name)
        
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Query collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Format results
        documents = []
        if results and results["documents"] and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                doc_dict = {
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {}
                }
                documents.append(doc_dict)
        
        return documents
        
    except Exception as e:
        print(f"⚠️  Error querying ChromaDB: {str(e)}")
        return []


def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible"""
    try:
        response = requests.get(f"{OLLAMA_API_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def generate_response_with_ollama(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 512
) -> str:
    """Generate response using Ollama"""
    
    # Check Ollama connection
    if not check_ollama_connection():
        raise Exception(
            f"❌ Ollama is not running at {OLLAMA_API_URL}\n"
            "Please start Ollama with: ollama serve"
        )
    
    # Prepare context from retrieved documents
    context = ""
    if retrieved_docs:
        context = "\n\n".join([
            f"Document: {doc['metadata'].get('source', 'Unknown')} (Page {doc['metadata'].get('page', 'N/A')})\n"
            f"Content: {doc['content']}"
            for doc in retrieved_docs
        ])
    
    # Build prompt
    system_prompt = """You are a helpful college assistant. Answer questions based on the provided documents.
If the answer is not in the documents, say "I don't have information about that in the provided documents."
Provide clear, concise answers with relevant details from the sources."""
    
    if context:
        prompt = f"""Based on the following documents:

{context}

Answer this question: {query}"""
    else:
        prompt = query
    
    try:
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_API_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": system_prompt,
                "stream": False,
                "temperature": temperature,
                "num_predict": max_tokens,
            },
            timeout=600
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No response generated")
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
            
    except requests.exceptions.Timeout:
        raise Exception("❌ Ollama request timed out. Please check if Ollama is running.")
    except Exception as e:
        raise Exception(f"❌ Error generating response: {str(e)}")


def save_chat_history(chat_history: List[Dict[str, Any]]) -> None:
    """Save chat history to file"""
    try:
        with open(CHAT_HISTORY_FILE, "w") as f:
            json.dump(chat_history, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️  Error saving chat history: {str(e)}")


def load_chat_history() -> List[Dict[str, Any]]:
    """Load chat history from file"""
    try:
        if Path(CHAT_HISTORY_FILE).exists():
            with open(CHAT_HISTORY_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️  Error loading chat history: {str(e)}")
    
    return []
