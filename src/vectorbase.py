from qdrant_client import QdrantClient
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings

class Vectorbase(Qdrant):

    def __init__(self, embed_model_name:str):
        
        # TODO: host locally?
        embeddings = HuggingFaceEmbeddings(
            model_name=embed_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )

        super().__init__(QdrantClient(), "Peter's useful notes", embeddings)


    # Return VectorStore initialized from documents and embeddings    
    def build_vectordb(self, docs):
        
        assert self.embeddings is not None, "Please initialize the embedding model first"

        vectorstore = Qdrant.from_documents(
            docs,
            self.embeddings,
            location=":memory:",  # Local mode with in-memory storage only
            collection_name="MD_files",
        )
        
        return vectorstore
    