# tools.py
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """
    Evaluate a simple math expression.
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"
