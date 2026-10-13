
import time

from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from app.core.config import get_settings

settings = get_settings()

_embeddings = None
_vectorstore = None

# BAAI/bge-m3 produces 1024-dimensional embeddings.
EMBEDDING_DIMENSIONS = {
    "BAAI/bge-m3": 1024,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}


def get_embedding_dimension(model_name: str | None = None) -> int:
    name = (model_name or settings.embedding_model or "").strip()

    if not name:
        raise RuntimeError("EMBEDDING_MODEL is not configured")

    normalized = name.lower()

    for model, dimension in EMBEDDING_DIMENSIONS.items():
        if normalized == model.lower():
            return dimension

    raise ValueError(
        f"Unsupported embedding model: {name}. "
        "Add its correct dimension to EMBEDDING_DIMENSIONS."
    )


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        model_name = settings.embedding_model or "BAAI/bge-m3"

        _embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    existing_names = pc.list_indexes().names()

    if index_name in existing_names:
        info = pc.describe_index(index_name)
        current_dimension = info.dimension

        if current_dimension != desired_dimension:
            raise ValueError(
                f"Pinecone index '{index_name}' has dimension "
                f"{current_dimension}, but model "
                f"'{settings.embedding_model}' requires "
                f"{desired_dimension}. Create a separate index "
                "with the correct dimension or deliberately "
                "recreate the existing index after backing up "
                "any vectors you need."
            )

        return pc.Index(index_name)

    pc.create_index(
        name=index_name,
        dimension=desired_dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    while not pc.describe_index(index_name).status.ready:
        time.sleep(1)

    return pc.Index(index_name)


def get_vectorstore():
    global _vectorstore

    if _vectorstore is None:
        index = ensure_index()

        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )

    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": settings.top_k}
    )


def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)
