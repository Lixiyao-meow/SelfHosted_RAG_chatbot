import os
import dotenv
from fastapi import FastAPI
import uvicorn

import markdown_loader as markdown_loader
import vectorbase as vectorbase

def safe_load_env(env_var: str):
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
db = vectorbase.Vectorbase(embed_model_name)
vectordb = db.build_vectordb(docs)

RAG_chatbot = FastAPI()

# query the vector database
@RAG_chatbot.get("/similarity_search/")
def query_database(query: str):
    answer = vectordb.similarity_search(query, k=4)
    return {"answer": answer}

if __name__ == "__main__":
    
    uvicorn.run(
        RAG_chatbot, 
        host=safe_load_env("HOST"), 
        port=int(safe_load_env("PORT"))
    )