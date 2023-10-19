import os
import dotenv
from fastapi import FastAPI
import uvicorn

import markdown_loader as markdown_loader
import vectorbase as vectorbase

dotenv.load_dotenv()
markdown_path = os.getenv("MARKDOWN_PATH")
embed_model_name = os.getenv("EMBED_MODEL_NAME")

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
        host=os.getenv("HOST"), 
        port=int(os.getenv("PORT"))
    )