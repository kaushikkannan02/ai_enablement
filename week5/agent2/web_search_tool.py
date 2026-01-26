from tavily import TavilyClient
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool



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
You are an IT-focused web search tool.

Purpose:
- Retrieve accurate, up-to-date IT information relevant to corporate environments.

When to use:
- Use this tool ONLY if the required information is not available in internal IT documentation.
- Prefer official vendor documentation, enterprise knowledge bases, or well-known technical sources.

Source preferences (in order):
1. Official vendor documentation (Microsoft, Apple, Cisco, Fortinet, Palo Alto, etc.)
2. Enterprise IT documentation sites
3. Well-established technical communities (Stack Overflow, Red Hat docs)

Avoid:
- Personal blogs
- Marketing pages
- Opinion-based articles
- Unverified forums

Output requirements:
- Return concise, factual information
- Include step-by-step instructions when applicable
- Clearly state if authoritative information cannot be found

External Information:
{formatted_results}
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


web_search_tool = Tool(
    name="external_web_research",
    func=web_research,
    description=(
        "Search external authoritative sources for IT-related information "
        "that is NOT available in the company's internal IT documentation. "
        "Use this tool only as a fallback when internal policies or procedures "
        "do not contain the required information. "
        "Preferred sources include official vendor documentation, "
        "enterprise IT knowledge bases, and well-established technical references. "
        "This tool should be used for topics such as supported operating systems, "
        "vendor-specific setup instructions, or technical standards."
    ),
)


