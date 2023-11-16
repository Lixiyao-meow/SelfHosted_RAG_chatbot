import dotenv
from fastapi import FastAPI
import uvicorn

from langchain.embeddings import HuggingFaceEmbeddings

import markdown_loader as markdown_loader
from database import local_database as local_database
from database import hosted_database as hosted_database
from generators.local_llm import LLM_Model
from generators.hosted_llm import HostedLlm
from generators.interfaces import EmbeddingModel, SimpleGenerativeModel
from settings import HostedLlmSettings, InMemoryEmbeddingSettings, InMemoryLlmSettings, Settings, load_settings

# Load environment variables from .env file
dotenv.load_dotenv()
app_settings: Settings = load_settings(Settings)

# Loading LLM model could be done lazily
# For simplicity we load it eagerly for now
if app_settings.HOSTED_LLM:
    # Use hosted LLM
    llm_settings = load_settings(HostedLlmSettings)
    generative_model = HostedLlm(
        llm_settings.LLM_API_BASE,
        llm_settings.LLM_API_KEY,
    )
else:    
    # Load settings and construct LLM model to be used in-memory
    llm_settings = load_settings(InMemoryLlmSettings)
    generative_model = LLM_Model(
        llm_settings.MODEL_PATH, 
        n_gpu_layers=llm_settings.N_GPU_LAYERS, 
        verbose=llm_settings.VERBOSE, 
        embedding=llm_settings.EMBEDDING
    )
    # If we allow to use LLM for embedding, we map its embedding function to the embedding interface
    if llm_settings.EMBEDDING:
        # Embedding via LLama.cpp model
        embedding = EmbeddingModel(generative_model.llm.embed)

if app_settings.HOSTED_EMBEDDING:
    raise Exception("Hosted embedding is not implemented yet")
else:
    # Use inmemory embedding
    embed_settings = load_settings(InMemoryEmbeddingSettings)
    # For simplicity use HuggingFace model for embedding
    tmp = HuggingFaceEmbeddings(
        model_name=embed_settings.EMBED_MODEL_NAME,
        model_kwargs={'device': embed_settings.EMBED_DEVICE},
        encode_kwargs={'normalize_embeddings': False}
    )
    embedding = EmbeddingModel(tmp.embed_query)


# Load markdown files
markdown_path: str = app_settings.MARKDOWN_PATH
loader = markdown_loader.MarkdownLoader(markdown_path)
docs = loader.load_batch()

# init vector database
vectordb = hosted_database.build_database(
    embedding, 
    [doc for doc in docs],
    database_url=app_settings.DATABASE_URL)

RAG_chatbot = FastAPI()

# query the vector database
@RAG_chatbot.get("/similarity_search/")
def query_database(query: str, k: int = 4):
    """
    Using query, retrieve relevant documents from the database
    """
    answer = vectordb.similarity_search_with_score(query, k=k)
    return {"answer": answer}

# This is would be used to inject the model into the API
# The simplicity is temporary, we will need to add more functionality here to decide what model to use
def get_model() -> SimpleGenerativeModel:
    return generative_model

@RAG_chatbot.get("/rag")
def generate_answer(query: str):
    """
    Using query, retrieve relevant documents from the database and generate answer via LLM
    """
    scored_docs = query_database(query, 4)["answer"]
    answer = get_model().RAG_QA_chain([pairs[0] for pairs in scored_docs], query)
    return {"answer": answer}

    
if __name__ == "__main__":
    
    uvicorn.run(
        RAG_chatbot, 
        host=app_settings.HOST, 
        port=app_settings.PORT,
    )