import os
import dotenv
from typing import List

from langchain.schema import Document
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain import hub
from langchain.chains.question_answering import load_qa_chain

class LLM_Model():
    
    def __init__(self, model_path:str, n_gpu_layers:int=0, n_batch:int=512, 
                 temperature:float=0.25, max_tokens:int=2000, n_ctx:int=2048, 
                 verbose:bool=True):
        
        self.model_path = model_path
        
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers, # TODO: make it work with GPU
            #n_batch=n_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
            verbose=verbose
        )

    def RAG_QA_chain(self, retrieved_docs: List[Document], query: str):
        
        assert self.llm is not None, "LLM is not initialized"

        rag_prompt = hub.pull("rlm/rag-prompt")
        chain = load_qa_chain(self.llm, chain_type="stuff", prompt=rag_prompt)
        chain({"input_documents": retrieved_docs, "question": query}, return_only_outputs=True)
        # how to output the answer?