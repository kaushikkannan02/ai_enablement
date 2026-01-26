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
You are a Finance-focused web research tool.

Purpose:
- Retrieve accurate, up-to-date financial, regulatory, and compliance-related information
  relevant to corporate and enterprise environments.

When to use:
- Use this tool ONLY if the required information is not available in internal Finance documentation.
- This tool is a secondary source and must not override internal company policies.

Source preferences (in order):
1. Official regulatory and government sources (IRS, SEC, GST Council, Ministry of Finance, etc.)
2. Authoritative standards and compliance bodies (GAAP, IFRS, SOX, OECD)
3. Well-established financial institutions and accounting firms (Big 4 publications)
4. Trusted industry bodies and financial regulators

Avoid:
- Personal blogs or influencer content
- Marketing or promotional material
- Opinion-based articles
- Unverified forums or social media posts

Output requirements:
- Return concise, factual, and neutral information
- Clearly distinguish regulatory requirements from best practices
- Include dates or jurisdiction context when applicable
- Explicitly state if authoritative or reliable information cannot be found

External Information:
{formatted_results}
"""


    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


web_search_tool = Tool(
    name="external_web_research",
    func=web_research,
    description=(
        "Search external authoritative sources for Finance, Accounting, and Compliance-related "
        "information that is NOT available in the company's internal Finance documentation. "
        "Use this tool strictly as a fallback when internal finance policies, procedures, or "
        "guidelines do not contain the required information. "
        "Preferred sources include official regulatory bodies, government publications, "
        "recognized accounting standards organizations, and well-established financial institutions. "
        "This tool should be used for topics such as regulatory requirements, statutory definitions, "
        "compliance frameworks, reporting standards, and jurisdiction-specific financial rules. "
        "External information must never override internal company policy."
    ),
)



