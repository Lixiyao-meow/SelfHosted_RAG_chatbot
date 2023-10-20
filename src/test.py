import os
import dotenv
from fastapi import FastAPI
import uvicorn

from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate

import markdown_loader as markdown_loader
import vectorbase as vectorbase
import llm_model as llm_model

dotenv.load_dotenv()


markdown_path = os.getenv("MARKDOWN_PATH")
embed_model_name = os.getenv("EMBED_MODEL_NAME")
model_path = os.getenv("MODEL_PATH") # TODO: change model cause current one is not good


# load markdown files
loader = markdown_loader.MarkdownLoader(markdown_path)
docs = loader.load_batch()

# init vector database
print("init vector database ... ")
db = vectorbase.Vectorbase(embed_model_name)
vectordb = db.build_vectordb(docs)

print("retrieve documents ... ")
query = "How to update Kubernetes?"
retrieved_docs = vectordb.similarity_search(query, k=1)

print("Answering question ... ")
llm = llm_model.LLM_Model(model_path, verbose=False)
print(llm.RAG_QA_chain(retrieved_docs, query))