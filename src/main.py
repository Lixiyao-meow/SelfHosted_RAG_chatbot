import os
import dotenv
from fastapi import FastAPI
import uvicorn

import markdown_loader as markdown_loader
import local_database as local_database
import hosted_database as hosted_database
#from llm_model import LLM_Model

def safe_load_env(env_var: str) -> str:
    r = os.getenv(env_var)
    if r is None:
        print(f"Please set the environment variable {env_var}")
        exit(1)
    return r

dotenv.load_dotenv()
markdown_path = safe_load_env("MARKDOWN_PATH")
embed_model_name = safe_load_env("EMBED_MODEL_NAME")

# load markdown files
loader = markdown_loader.MarkdownLoader(markdown_path)
docs = loader.load_batch()

# init vector database
#db = local_database.Vectorbase(embed_model_name)
#vectordb: Qdrant = db.build_vectordb(docs)

vectordb = hosted_database.build_database(embed_model_name, 
                                          docs, 
                                          database_url="http://localhost")

RAG_chatbot = FastAPI()

# query the vector database
@RAG_chatbot.get("/similarity_search/")
def query_database(query: str):
    answer = vectordb.similarity_search(query, k=4)
    return {"answer": answer}

'''
lazy_model: LLM_Model | None = None
def get_model() -> LLM_Model:
    global lazy_model
    if lazy_model is None:
        model_path = safe_load_env("MODEL_PATH")
        lazy_model = LLM_Model(model_path, n_gpu_layers=0, verbose=False)
    return lazy_model

@RAG_chatbot.get("/rag")
def generate_answer(query: str):
    docs = vectordb.similarity_search(query, k=4)
    answer = get_model().RAG_QA_chain(docs, query)
    return {"answer": answer}
'''
if __name__ == "__main__":
    
    uvicorn.run(
        RAG_chatbot, 
        host=safe_load_env("HOST"), 
        port=int(safe_load_env("PORT"))
    )