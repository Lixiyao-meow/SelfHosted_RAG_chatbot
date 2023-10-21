from typing import List

from langchain.schema import Document
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks import StdOutCallbackHandler

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
            callback_manager=CallbackManager([StdOutCallbackHandler()]),
            verbose=verbose,
            stop = ["<|im_end|>", "<|im_start|>"],
        ) # type: ignore

    def fill_prompt(self, context: str, prompt: str):
        return f"""
        <|im_start|>system
        You are a helpful assistant. You are helping a user with a question.
        Answer in a concise way in a few sentences.
        Use the following context to answer the user's question.
        If the given given context does not have the information to answer the question, you should answer "I don't know" and don't say anything else.
        Context:
        {context}<|im_end|>
        <|im_start|>user
        {prompt}<|im_end|>
        <|im_start|>assistant
        """

    def RAG_QA_chain(self, retrieved_docs: List[Document], query: str) -> str:

        assert self.llm is not None, "LLM is not initialized"

        context: str = "\n".join(doc.page_content for doc in retrieved_docs)
        result: str = self.llm(self.fill_prompt(context, query))
        
        return result