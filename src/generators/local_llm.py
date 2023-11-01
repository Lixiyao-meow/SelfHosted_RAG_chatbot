from typing import List

from langchain.schema import Document
from llama_cpp import Completion, Llama

class PromptTemplates():

    @staticmethod
    def marx(prompt: str, user: str = "HUMAN", response: bool = False):
        return f"""### {user}:\n{prompt}\n\n""" + "### RESPONSE:\n" if response else ""
    
    @staticmethod
    def tiny_llama(prompt: str, user: str = "HUMAN"):
        return f"""<|im_start|>{user}\n{prompt}"""


class LLM_Model():
    def __init__(self, 
                 model_path:str, 
                 n_gpu_layers:int=0, 
                 n_batch:int=512, 
                 temperature:float=0.25, 
                 max_tokens:int=2000, 
                 n_ctx:int=2048,
                 embedding: bool = False,
                 verbose:bool=True):
        
        self.model_path = model_path

        self.llm = Llama(
            model_path=model_path,
            n_gpu_layers=n_gpu_layers, # TODO: make it work with GPU
            #n_batch=n_batch,
            temperature=temperature,
            max_tokens=max_tokens,
            n_ctx=n_ctx,
            verbose=verbose,
            stop = ["<|im_end|>", "<|im_start|>"],
            embedding=embedding,
            stream = False
        ) # type: ignore


        # self.llm = LlamaCpp(
        #     model_path=model_path,
        #     n_gpu_layers=n_gpu_layers, # TODO: make it work with GPU
        #     #n_batch=n_batch,
        #     temperature=temperature,
        #     max_tokens=max_tokens,
        #     n_ctx=n_ctx,
        #     callback_manager=CallbackManager([StdOutCallbackHandler()]),
        #     verbose=verbose,
        #     stop = ["<|im_end|>", "<|im_start|>"],
        # ) # type: ignore

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
    
    system_message = """You are a helpful assistant. You are helping a user with a question.
        Answer in a concise way in a few sentences.
        Use the following context to answer the user's question.
        If the given given context does not have the information to answer the question, you should answer "I don't know" and don't say anything else."""

    def RAG_QA_chain(self, retrieved_docs: List[Document], query: str) -> str:

        assert self.llm is not None, "LLM is not initialized"

        context: str = "\n".join(doc.page_content for doc in retrieved_docs)
        # We ignore the type because it is entirely dependent on streaming
        msg = PromptTemplates.marx(self.system_message + "\n" + context, "SYSTEM") + PromptTemplates.marx(query, "HUMAN", True)
        result: Completion = self.llm.create_completion(msg, temperature=0.1) # type: ignore

        # return generated text
        return result["choices"][0]["text"]