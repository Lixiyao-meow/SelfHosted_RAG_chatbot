from typing import Any, Callable, Iterable, List

from llama_cpp import ChatCompletionMessage, ChatCompletionRequestMessage, ChatCompletionResponseMessage
from llama_cpp import llama_chat_format
from llama_cpp.llama_chat_format import ChatFormatterResponse

class TinyLlamaFormatter(llama_chat_format.ChatFormatter):
    _roles: dict[str, str] = {
        "user": "<|im_start|>user",
        "assistant": "<|im_start|>assisant",
        "system": "<|im_start|>system",
        "function": "<|im_start|>function"
    }

    def marx_message(self, msg: ChatCompletionRequestMessage | ChatCompletionResponseMessage) -> str:
        role = self._roles.get(msg["role"], None)
        if role is None:
            raise Exception("Unknown role")
        return f"""{role}\n{msg["content"]}<|im_end|>"""

    def call(self, messages: List[ChatCompletionRequestMessage | ChatCompletionResponseMessage], **kwargs: Any) -> ChatFormatterResponse:
        return ChatFormatterResponse(
            prompt = "\n".join(self.marx_message(msg) for msg in messages) + "\n" + self._roles["assistant"],
            stop = ["<|im_start|>", "<|im_end|>"] + [x for x in self._roles.values()],
            **kwargs
        )
    def __call__(self, messages: List[ChatCompletionRequestMessage | ChatCompletionResponseMessage], **kwargs: Any) -> ChatFormatterResponse:
        return self.call(messages, **kwargs)