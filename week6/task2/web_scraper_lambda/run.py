import boto3


REGION = "us-east-1"
AGENT_ID = "IWLCXHVXEK"
AGENT_ALIAS_ID = "TSTALIASID"   # <-- replace with real alias id
SESSION_ID = "webscrape-session-1"

def run_agent(prompt: str):
    client = boto3.client(
        "bedrock-agent-runtime",
        region_name=REGION
    )

    response = client.invoke_agent(
        agentId=AGENT_ID,
        agentAliasId=AGENT_ALIAS_ID,
        sessionId=SESSION_ID,
        inputText=prompt
    )

    print("\n--- AGENT RESPONSE ---\n")

    # Stream the response
    for event in response["completion"]:
        if "chunk" in event:
            chunk = event["chunk"]
            if "bytes" in chunk:
                print(chunk["bytes"].decode("utf-8"), end="")

    print("\n\n--- END ---")


if __name__ == "__main__":
    run_agent("Crawl this URL: https://en.wikipedia.org/wiki/Web_scraping")




