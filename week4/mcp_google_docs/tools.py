from .google_docs_client import search_insurance_docs

TOOLS = {
    "search_google_docs": {
        "name": "search_google_docs",
        "description": (
            "Search Presidio insurance-related Google Docs "
            "and return relevant policy or benefits information"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Insurance-related search query"
                }
            },
            "required": ["query"]
        }
    }
}


def run_tool(tool_name: str, args: dict) -> str:
    if tool_name == "search_google_docs":
        docs = search_insurance_docs(args["query"])

        if not docs:
            return "No insurance-related documents found."

        return "\n\n".join(
            f"Title: {d['title']}\n{d['content']}"
            for d in docs
        )

    raise ValueError("Unknown tool")
