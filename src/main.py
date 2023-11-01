import dotenv
from fastapi import FastAPI
import uvicorn

from langchain.embeddings import HuggingFaceEmbeddings

import markdown_loader as markdown_loader
import local_database as local_database
import hosted_database as hosted_database
from generators.local_llm import LLM_Model
from generators.interfaces import EmbeddingModel, SimpleGenerativeModel
from settings import Settings, load_settings

# Load environment variables from .env file
dotenv.load_dotenv()
app_settings: Settings = load_settings()

markdown_path: str = app_settings.MARKDOWN_PATH
embed_model_name: str = app_settings.EMBED_MODEL_NAME

# load markdown files
loader = markdown_loader.MarkdownLoader(markdown_path)
docs = loader.load_batch()

# We should do this conditionally, but for now we just do it here
generative_model = LLM_Model(app_settings.MODEL_PATH, n_gpu_layers=40, verbose=True, embedding=True)
# Embedding via LLama.cpp model
embedding = EmbeddingModel(generative_model.llm.embed)

# Use HuggingFace model for embedding
hf_embedding = HuggingFaceEmbeddings(
    model_name=embed_model_name,
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': False}
)
embedding = EmbeddingModel(hf_embedding.embed_query)

# init vector database
vectordb = hosted_database.build_database(
    embedding, 
    docs, 
    database_url="http://localhost")

RAG_chatbot = FastAPI()

# query the vector database
@RAG_chatbot.get("/similarity_search/")
def query_database(query: str):
    answer = vectordb.similarity_search(query, k=4)
    return {"answer": answer}

# This is would be used to inject the model into the API
# The simplicity is temporary, we will need to add more functionality here to decide what model to use
def get_model() -> SimpleGenerativeModel:
    return generative_model

@RAG_chatbot.get("/rag")
def generate_answer(query: str):
    docs = vectordb.similarity_search(query, k=4)
    answer = get_model().RAG_QA_chain(docs, query)
    return {"answer": answer}

    
if __name__ == "__main__":
    
    uvicorn.run(
        RAG_chatbot, 
        host=app_settings.HOST, 
        port=app_settings.PORT,
    )