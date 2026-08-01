#!/usr/bin/env python3
"""
Standalone script to ingest PDF documents into ChromaDB.
Usage: python ingest.py
"""

from pathlib import Path
from dotenv import load_dotenv
import sys

from utils import (
    initialize_chromadb,
    initialize_embeddings,
    load_and_split_documents,
    add_documents_to_chromadb,
)


def main():
    """Main ingestion pipeline"""

    # Load environment variables
    load_dotenv()

    # Path to folder containing PDFs
    college_data_dir = Path("college_data")

    # Create directory if it doesn't exist
    college_data_dir.mkdir(exist_ok=True)

    # Check for PDF files
    pdf_files = list(college_data_dir.glob("*.pdf"))

    if not pdf_files:
        print("❌ No PDF files found in college_data/ directory")
        print("📋 Please add PDF files to the college_data/ folder and run this script again.")
        return

    print(f"✅ Found {len(pdf_files)} PDF file(s)")

    try:
        # Initialize embeddings
        print("\n🧠 Initializing embeddings...")
        embeddings = initialize_embeddings()
        print("✅ Embeddings initialized")

        # Initialize ChromaDB
        print("\n📦 Initializing ChromaDB...")
        vector_db = initialize_chromadb()
        print("✅ ChromaDB initialized")

        # Load and split documents
        print("\n📄 Loading and splitting documents...")
        file_paths = [str(f) for f in pdf_files]
        documents = load_and_split_documents(file_paths)
        print(f"✅ Loaded {len(documents)} document chunks")

        # Add to ChromaDB
        print("\n💾 Adding documents to database...")
        add_documents_to_chromadb(vector_db, documents, embeddings)
        print("✅ Documents stored in ChromaDB")

        print("\n🎉 Ingestion complete!")
        print("Run:")
        print("streamlit run app.py")

    except Exception as e:
        print(f"\n❌ Error during ingestion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()