from typing import List
from langchain.schema import Document

from langchain.vectorstores.qdrant import Qdrant
from langchain.schema.embeddings import Embeddings

# Dependency inject embedding
def build_database(embedding: Embeddings, docs: List[Document], database_url:str="localhost", database_port:int=6333) -> Qdrant:
    return Qdrant.from_documents(
        docs,
        embedding,
        url=database_url,
        port = database_port,
        #prefer_grpc=True,
        collection_name="Peter's useful notes",
        force_recreate=True,
        distance_func = "Cosine",
    )