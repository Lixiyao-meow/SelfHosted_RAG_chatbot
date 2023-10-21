from typing import List
from langchain.schema import Document

from langchain.vectorstores.qdrant import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

def build_database(embed_model_name:str, docs: List[Document], database_url:str="localhost", database_port:int=6333):
        
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': False}
    )
        
    return Qdrant.from_documents(
        docs,
        embeddings,
        url=database_url,
        #prefer_grpc=True,
        collection_name="Peter's useful notes",
        force_recreate=True
    )