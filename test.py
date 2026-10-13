# from app.core.config import get_settings

# settings = get_settings()

# print(f"Open ai api key: {settings.deepseek_api_key}")

# from app.services.ingestion import load_file,chunk_documents
# from pathlib import Path

# docs = load_file(Path("data/sample_kb/company_hr_handbook.md"))

# chunked_docs = chunk_documents(docs)

# print(chunked_docs)

from app.rag.vectorstore import get_embeddings

embeddings = get_embeddings()

vector = embeddings.embed_query(
    "How many annual leave days do employees receive?"
)

print("Embedding dimensions:", len(vector))
print("First five values:", vector[:5])