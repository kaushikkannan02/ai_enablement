# agent.py
from langchain_aws import ChatBedrock
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from tools import calculator

from langfuse.langchain import CallbackHandler
from nemoguardrails import LLMRails, RailsConfig

# ---------------------------
# Langfuse
# ---------------------------
langfuse_handler = CallbackHandler()

# ---------------------------
# Bedrock Claude
# ---------------------------
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={"temperature": 0, "max_tokens": 300},
)

# ---------------------------
# LangChain Agent
# ---------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use tools when necessary."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [calculator]

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    callbacks=[langfuse_handler],
)

# ---------------------------
# NeMo Guardrails
# ---------------------------
rails = LLMRails(RailsConfig.from_path("./guardrails"))

def guarded_agent_invoke(user_input: str):
    # Ask NeMo to evaluate the input
    result = rails.generate(
        messages=[{"role": "user", "content": user_input}]
    )

    # If NeMo blocked it, it already responded
    if result["role"] == "assistant" and result["content"] != user_input:
        return result["content"]

    # Otherwise run LangChain agent
    response = agent_executor.invoke({"input": user_input})
    output = response["output"]

    # Validate output
    result = rails.generate(
        messages=[{"role": "assistant", "content": output}]
    )

    if result["content"] != output:
        return result["content"]

    return output

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    print(guarded_agent_invoke("What is (23 * 7) + 5?"))


# export LANGFUSE_SECRET_KEY="sk-lf-1e177f0d-3f90-447f-bf19-0ee564dde928"
# export LANGFUSE_PUBLIC_KEY="pk-lf-b23c56b4-3f97-43a3-a865-2c3e71a9890a"
# export LANGFUSE_BASE_URL="https://cloud.langfuse.com"
