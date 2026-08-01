import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
import json
from datetime import datetime

from utils import (
    initialize_chromadb,
    initialize_embeddings,
    load_and_split_documents,
    add_documents_to_chromadb,
    query_chromadb,
    generate_response_with_ollama,
    save_chat_history,
    load_chat_history,
)

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="College RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .chat-message {
        padding: 1rem;
        margin-bottom: 1rem;
        border-radius: 0.5rem;
        display: flex;
        gap: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
        margin-left: 2rem;
    }
    .assistant-message {
        background-color: #f5f5f5;
        margin-right: 2rem;
    }
    .source-container {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .source-title {
        font-weight: bold;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = load_chat_history()

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = initialize_embeddings()

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

# Sidebar
with st.sidebar:
    st.title("📚 College RAG Assistant")
    
    # Initialize vector database
    if st.button("🔄 Initialize Database", key="init_db"):
        with st.spinner("Initializing ChromaDB..."):
            st.session_state.vector_db = initialize_chromadb()
            st.success("✅ Database initialized!")
    
    st.divider()
    
    # PDF Upload Section
    st.subheader("📄 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        key="pdf_uploader"
    )
    
    if uploaded_files and st.session_state.vector_db:
        if st.button("📤 Process & Store Documents"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Save uploaded files temporarily
                temp_dir = Path("temp_uploads")
                temp_dir.mkdir(exist_ok=True)
                
                file_paths = []
                for uploaded_file in uploaded_files:
                    file_path = temp_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(str(file_path))
                
                # Load and split documents
                status_text.text("Loading and splitting documents...")
                documents = load_and_split_documents(file_paths)
                
                # Add to ChromaDB
                status_text.text("Adding documents to database...")
                add_documents_to_chromadb(
                    st.session_state.vector_db,
                    documents,
                    st.session_state.embeddings
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Documents processed successfully!")
                st.success(f"Processed {len(uploaded_files)} file(s) with {len(documents)} chunks!")
                st.session_state.documents_loaded = True
                
                # Cleanup
                for file_path in file_paths:
                    os.remove(file_path)
                temp_dir.rmdir()
                
            except Exception as e:
                st.error(f"❌ Error processing documents: {str(e)}")
    
    elif uploaded_files and not st.session_state.vector_db:
        st.warning("⚠️ Please initialize the database first!")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    
    temperature = st.slider(
        "Model Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher = more creative, Lower = more factual"
    )
    
    top_k = st.slider(
        "Retrieve Top K Documents",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Number of source documents to retrieve"
    )
    
    st.divider()
    
    # Clear chat history
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        save_chat_history([])
        st.success("Chat history cleared!")
    
    # Info
    st.divider()
    st.markdown("""
    ### ℹ️ About
    **College RAG Assistant** uses:
    - 📦 ChromaDB for vector storage
    - 🤖 Llama 3:8B (Ollama) for responses
    - 🧠 Hugging Face embeddings
    - 📄 LangChain for document processing
    """)

# Main content
st.title("📚 College RAG Assistant")

if not st.session_state.documents_loaded:
    st.info("👈 Please upload and process documents using the sidebar to get started!")
else:
    # Chat interface
    st.subheader("💬 Chat")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <div style="flex: 1;">
                        <strong>You:</strong><br>
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <div style="flex: 1;">
                        <strong>Assistant:</strong><br>
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if "sources" in message and message["sources"]:
                    st.markdown(f"""
                    <div class="source-container">
                        <div class="source-title">📋 Sources:</div>
                        {message['sources']}
                    </div>
                    """, unsafe_allow_html=True)
    
    # Input area
    st.divider()
    user_input = st.text_input(
        "Ask a question about your documents...",
        placeholder="What is the admission process?",
        key="user_input"
    )
    
    if user_input:
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now().isoformat()
        })
        
        with st.spinner("🤔 Thinking..."):
            try:
                # Retrieve relevant documents
                retrieved_docs = query_chromadb(
                    st.session_state.vector_db,
                    user_input,
                    st.session_state.embeddings,
                    k=top_k
                )
                
                # Generate response
                response = generate_response_with_ollama(
                    user_input,
                    retrieved_docs,
                    temperature=temperature
                )
                
                # Format sources
                sources_html = ""
                if retrieved_docs:
                    for i, doc in enumerate(retrieved_docs, 1):
                        source = doc.get("metadata", {}).get("source", "Unknown")
                        page = doc.get("metadata", {}).get("page", "N/A")
                        sources_html += f"<div>• <strong>{i}.</strong> {source} (Page {page})</div>"
                
                # Add assistant message to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response,
                    "sources": sources_html,
                    "timestamp": datetime.now().isoformat()
                })
                
                # Save chat history
                save_chat_history(st.session_state.chat_history)
                
                # Rerun to display new messages
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
