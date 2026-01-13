import chromadb
import os
from ingestion import read_document, split_text
from chromadb.utils import embedding_functions

# Initialize ChromaDB client with persistence
client = chromadb.PersistentClient(path="chroma_db")

# Configure sentence transformer embeddings
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Create or get existing collection
client.delete_collection("documents_collection")
collection = client.get_or_create_collection(
    name="documents_collection",
    embedding_function=sentence_transformer_ef
)


def process_document(file_path: str):
    """Process a single document and prepare it for ChromaDB"""
    try:
        # Read the document
        content = read_document(file_path)

        # Split into chunks
        chunks = split_text(content)

        # Prepare metadata
        file_name = os.path.basename(file_path)
        metadatas = [{"source": file_name, "chunk": i} for i in range(len(chunks))]
        ids = [f"{file_name}_chunk_{i}" for i in range(len(chunks))]

        return ids, chunks, metadatas
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []