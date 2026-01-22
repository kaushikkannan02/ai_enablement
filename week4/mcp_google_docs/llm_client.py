# from langchain_aws import ChatBedrock
# from langchain_core.prompts import PromptTemplate
# from .mcp_tool_caller import fetch_full_docs_from_mcp
# from langchain_core.tools import Tool
# llm = ChatBedrock(
#     model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#     region_name="us-east-1",
#     model_kwargs={"temperature": 0.2, "max_tokens": 800}
# )

# prompt = PromptTemplate(
#     input_variables=["context", "question"],
#     template="""
# You are an insurance domain expert.

# Answer the question using ONLY the information in the context below.
# If the answer is not present in the context, say:
# "I could not find this information in the provided documents."

# Context:
# ====================
# {context}
# ====================

# Question:
# {question}

# Answer:
# """
# )

# def answer_query_using_full_docs(query: str) -> str:
#     docs_context = fetch_full_docs_from_mcp(query)

#     if not docs_context.strip():
#         return "No relevant documents were found."

#     final_prompt = prompt.format(
#         context=docs_context,
#         question=query
#     )
#     print("MCP Calling:::")
#     print(final_prompt)

#     response = llm.invoke(final_prompt)
#     return response.content


# insurance_tool = Tool(
#     name="InsurancePolicyTool",
#     description="Answer questions about Presidio's insurance policies",
#     func=answer_query_using_full_docs  # ← Use the LLM function!
# )
from langchain_aws import ChatBedrock
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from .mcp_tool_caller import fetch_full_docs_from_mcp

llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0.2, "max_tokens": 800}
)

# Clean prompt that returns factual answers
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are an insurance policy expert. Answer the question using ONLY the context below.

Be direct and factual. Do not use conversational phrases like "Here is a summary" or "Based on the documents".
Just state the facts clearly in a structured way.

If the information is not in the context, respond with:
"This information is not available in the insurance policy documents."

Context:
{context}

Question: {question}

Answer:"""
)

def answer_insurance_query(query: str) -> str:
    """Fetch docs from MCP and answer with LLM"""
    docs_context = fetch_full_docs_from_mcp(query)
    
    if not docs_context.strip():
        return "No insurance policy documents found."
    
    final_prompt = prompt.format(
        context=docs_context,
        question=query
    )
    
    response = llm.invoke(final_prompt)
    return response.content

# ✅ Tool that uses the LLM
insurance_tool = Tool(
    name="insurance_policy_search",
    description="Search Presidio's insurance policies for information about coverage, limits, premiums, claims procedures, etc.",
    func=answer_insurance_query
)