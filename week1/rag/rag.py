from llama import generate_response, contextualize_query
from session import add_message, format_history_for_prompt
from retrieval import get_context_with_sources, semantic_search
def rag_query(collection, query: str, n_chunks: int = 2):
    """Perform RAG query: retrieve relevant chunks and generate answer"""
    # Get relevant chunks
    results = semantic_search(collection, query, n_chunks)
    context, sources = get_context_with_sources(results)

    # Generate response
    response = generate_response(query, context)

    return response, sources

def conversational_rag_query(
    collection,
    query: str,
    session_id: str,
    n_chunks: int = 5
):
    """Perform RAG query with conversation history"""
    # Get only dialogue history
    conversation_history = format_history_for_prompt(session_id)

    # Handle follow-up questions
    query = contextualize_query(query, conversation_history)

    # Get only **current relevant chunks** from ChromaDB
    results = semantic_search(collection, query, n_chunks)
    context, sources = get_context_with_sources(results)
    print("Sources:", sources)

    # Generate response using **current context + conversation history**
    response = generate_response(query, context, conversation_history)

    # Store only user/assistant messages
    add_message(session_id, "user", query)
    add_message(session_id, "assistant", response)

    return response, sources
