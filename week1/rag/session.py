import uuid
from datetime import datetime
import json

# In-memory conversation store
conversations = {}

def create_session():
    """Create a new conversation session"""
    session_id = str(uuid.uuid4())
    conversations[session_id] = []
    return session_id

def add_message(session_id: str, role: str, content: str):
    """Add a message to the conversation history"""
    if session_id not in conversations:
        conversations[session_id] = []

    conversations[session_id].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    })

def get_conversation_history(session_id: str, max_messages: int = None):
    """Get conversation history for a session"""
    if session_id not in conversations:
        return []

    history = conversations[session_id]
    if max_messages:
        history = history[-max_messages:]

    return history


def format_history_for_prompt(session_id: str, max_messages: int = 5):
    """Format conversation history for inclusion in prompts"""
    history = get_conversation_history(session_id, max_messages)
    formatted_history = ""

    for msg in history:
        role = "Human" if msg["role"] == "user" else "Assistant"
        formatted_history += f"{role}: {msg['content']}\n\n"

    return formatted_history.strip()

