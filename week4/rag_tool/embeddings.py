from pathlib import Path

from langchain_aws import BedrockEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ✅ Resolve path relative to THIS file
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
