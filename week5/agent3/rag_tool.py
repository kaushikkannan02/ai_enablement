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
HR_POLICY_PATH = BASE_DIR / "finance_internal_docs.txt"

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
    """
You are a document retrieval and extraction tool for internal Finance documentation.

Purpose:
- Retrieve official, organization-approved Finance policies, procedures, and financial guidelines.
- Internal Finance documentation is the single source of truth for all finance-related information.

Usage rules:
- This tool must be used FIRST for all Finance-related queries involving:
  - Expense reimbursement policies
  - Payroll, compensation, and benefits
  - Procurement, vendor payments, and invoicing
  - Budgeting, approvals, and financial controls
  - Tax, audit, compliance, and regulatory requirements

Instructions:
- Use the provided context as the retrieved internal Finance documentation.
- Identify the sections that directly answer the user question.
- Extract information exactly as stated in the document.
- Do NOT infer, interpret, summarize, or add new information.
- Do NOT include recommendations unless explicitly present in the document.

If the context does not contain the required information:
- State clearly and exactly:
  "The requested information is not available in the internal Finance documentation."

Context:
{context}

User Question:
{question}

Response:
- Provide only extracted factual information from the context.
- Maintain original wording from the document wherever possible.
- If unavailable, respond with the unavailability statement exactly.
"""
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


finance_tool = Tool(
    name="finance_internal_policy_search",
    description=(
        "Search the company's internal Finance documentation to retrieve official, "
        "organization-approved information about Finance policies, procedures, and controls. "
        "Use this tool for questions related to expense reimbursement, payroll and compensation, "
        "procurement and vendor payments, budgeting and approvals, audits, tax, compliance, "
        "and other internal financial processes. "
        "This is the authoritative source for internal Finance information."
    ),
    func=lambda q: rag_chain.invoke(q)
)
