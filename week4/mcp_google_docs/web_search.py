from tavily import TavilyClient
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool
import sys


def format_web_results(response):
    summaries = []
    for r in response.get("results", []):
        summaries.append(
            f"Title: {r['title']}\n"
            f"Summary: {r['content']}\n"
            f"Source: {r['url']}\n"
        )
    return "\n---\n".join(summaries)


def web_research(user_query: str) -> str:
    client = TavilyClient()

    llm = ChatBedrock(
        model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        temperature=0
    )

    search_response = client.search(
        query=user_query,
        max_results=5,
        search_depth="advanced"
    )

    formatted_results = format_web_results(search_response)

    prompt = f"""
You are an internal research assistant for Presidio.

Instructions:
- Analyze external industry information
- Extract benchmarks, trends, or regulatory updates
- Keep the response concise and actionable
- Clearly state this is external data
- Cite sources

External Information:
{formatted_results}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def main():
    print("\nPresidio – External Industry Research CLI")
    print("Type your question and press Enter\n")

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Query: ").strip()

    if not query:
        print("❌ Query cannot be empty")
        sys.exit(1)

    print("\n🔍 Researching...\n")
    answer = web_research(query)

    print("📊 Result:\n")
    print(answer)


web_search_tool = Tool(
    name="external_web_research",
    func=web_research,
    description=(
        "Use this tool to fetch external industry benchmarks, "
        "market trends, regulatory updates, and compliance information. "
        "This tool should be used ONLY when the question requires "
        "information outside Presidio internal documents."
    ),
)
