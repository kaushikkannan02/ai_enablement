# from langchain_aws import ChatBedrock
# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from .embeddings import retriever
# from langchain_core.tools import Tool


# llm = ChatBedrock(
#     model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#     model_kwargs={
#         "temperature": 0.2,
#         "max_tokens": 500
#     }
# )

# prompt = PromptTemplate.from_template(
#     """You are an HR policies expert.

# Answer the question using ONLY the information in the context below.
# If the answer is not present in the context, say:
# "I could not find this information in the provided documents."

# Context:
# {context}

# Question:
# {question}

# Answer:
# """
# )

# rag_chain = (
#     {
#         "context": retriever,
#         "question": RunnablePassthrough()
#     }
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# def hr_query_with_llm(query: str) -> str:
#     """Call the full RAG chain that includes LLM"""
#     return rag_chain.invoke(query)

# hr_tool = Tool(
#     name="HRPolicyTool",
#     description="Answer questions about HR policies using company documents",
#     func=hr_query_with_llm  # ← Now returns final answer, not raw chunks
# )
from pathlib import Path
from langchain_aws import BedrockEmbeddings, ChatBedrock
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import Tool

# Setup (same as before)
BASE_DIR = Path(__file__).parent
HR_POLICY_PATH = BASE_DIR / "hr_policy.txt"

loader = TextLoader(str(HR_POLICY_PATH))
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(docs)

embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v1"
)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=str(BASE_DIR / "chroma_db"),
    collection_name="company_docs"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# LLM for RAG
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    model_kwargs={
        "temperature": 0.2,
        "max_tokens": 500
    }
)

# Prompt that returns clean, factual answers
prompt = PromptTemplate.from_template(
    """You are an HR policy expert. Answer the question using ONLY the context below.

Be direct and factual. Do not use phrases like "Based on the documents" or "According to the policy".
Just state the facts clearly.

If the information is not in the context, respond with:
"This information is not available in the HR policy documents."

Context:
{context}

Question: {question}

Answer:"""
)

# Complete RAG chain
rag_chain = (
    {
        "context": retriever,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | StrOutputParser()
)


hr_tool = Tool(
    name="hr_policy_search",
    description="Search Presidio's HR policies for information about vacation, sick leave, benefits, work hours, etc.",
    func=lambda q: rag_chain.invoke(q)
)