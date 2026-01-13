import requests
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"


def get_prompt(context, conversation_history, query):
  prompt = f"""Based on the following context and conversation history, please provide a relevant and contextual response.
    If the answer cannot be derived from the context, only use the conversation history or say "I cannot answer this based on the provided information."

    Context from documents:
    {context}

    Previous conversation:
    {conversation_history}

    Human: {query}

    Assistant:"""
  return prompt


def generate_response(query: str, context: str, conversation_history: str = ""):
    """Generate a response using LLaMA via Ollama"""
    prompt = get_prompt(context, conversation_history, query)

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "max_tokens": 500  # or num_predict: 500 depending on version
                }
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json()["response"]

    except Exception as e:
        return f"Error generating response: {str(e)}"
    

def contextualize_query(query: str, conversation_history: str):
    """
    Convert follow-up questions into standalone queries using LLaMA via Ollama.
    """
    contextualize_prompt = f"""
Given a chat history and the latest user question which might reference
context in the chat history, formulate a standalone question which can
be understood without the chat history. Do NOT answer the question,
just reformulate it if needed and otherwise return it as is.

Chat history:
{conversation_history}

Question:
{query}

Standalone question:
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "prompt": contextualize_prompt,
                "stream": False,
                "options": {
                    "temperature": 0,
                    "num_predict": 150  # enough tokens for reformulation
                }
            },
            timeout=60
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        print(f"Error contextualizing query: {str(e)}")
        return query  # fallback