import os
from embedding import process_document, collection
from rag import rag_query, conversational_rag_query
from session import create_session
DATA_DIR = "data"  # folder containing .txt files


def ingest_documents():
    """Load and index all TXT documents"""
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith(".txt"):
            continue

        file_path = os.path.join(DATA_DIR, filename)
        print(f"Ingesting: {filename}")

        ids, chunks, metadatas = process_document(file_path)

        if chunks:
            collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )


def main():
    print("=== RAG without LangChain (LLaMA + Chroma) ===\n")

    # Step 1: Ingest documents (run once)
    ingest_documents()

    # Step 2: Interactive query loop
    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        
        session_id = create_session()
        answer, sources = conversational_rag_query(
            collection,
            query,
            session_id
        )

        print("\nAnswer:\n" + "-" * 50)
        print(answer)

        print("\nSources:")
        for source in sources:
            print(f"- {source}")


if __name__ == "__main__":
    main()
