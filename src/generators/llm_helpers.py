from typing import Any, Callable, Iterable, List

from llama_cpp import ChatCompletionMessage, ChatCompletionRequestMessage, ChatCompletionResponseMessage
from llama_cpp import llama_chat_format
from llama_cpp.llama_chat_format import ChatFormatterResponse

class PromptTemplates():

    @staticmethod
    def templated(message: Iterable[ChatCompletionMessage], template: Callable[[ChatCompletionMessage], str], ending: str) -> str:
        """
        Convert a list of ChatCompletionMessage into a string that will be used as prompt
        """
        return "".join(template(msg) for msg in message) + ending


    @staticmethod
    def marx(prompt: str, user: str = "HUMAN", response: bool = False):
        return f"""### {user}:\n{prompt}\n\n""" + "### RESPONSE:\n" if response else ""
    
    @staticmethod
    def tiny_llama(prompt: str, user: str = "HUMAN"):
        return f"""<|im_start|>{user}\n{prompt}"""