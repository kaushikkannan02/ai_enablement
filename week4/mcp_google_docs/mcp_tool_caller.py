import requests

MCP_SERVER_URL = "http://localhost:3333"

def fetch_full_docs_from_mcp(query: str) -> str:
    response = requests.post(
        f"{MCP_SERVER_URL}/tools/search_google_docs",
        json={"arguments": {"query": "insurance"}},
        timeout=60
    )
    response.raise_for_status()
    return response.json()["result"]
