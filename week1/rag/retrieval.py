

def semantic_search(collection, query: str, n_results: int = 2):
    """Perform semantic search on the collection"""
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def get_context_with_sources(results):
    """Extract context and source information from search results"""
    # Combine document chunks into a single context
    context = "\n\n".join(results['documents'][0])

    # Format sources with metadata
    sources = [
        f"{meta['source']} (chunk {meta['chunk']})" 
        for meta in results['metadatas'][0]
    ]

    return context, sources

def print_search_results(results):
    """Print formatted search results"""
    print("\nSearch Results:\n" + "-" * 50)

    for i in range(len(results['documents'][0])):
        doc = results['documents'][0][i]
        meta = results['metadatas'][0][i]
        distance = results['distances'][0][i]

        print(f"\nResult {i + 1}")
        print(f"Source: {meta['source']}, Chunk {meta['chunk']}")
        print(f"Distance: {distance}")
        print(f"Content: {doc}\n")