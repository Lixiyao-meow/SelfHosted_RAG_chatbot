from typing import Any, Callable, Iterable, List

from llama_cpp import ChatCompletionMessage, ChatCompletionRequestMessage, ChatCompletionResponseMessage
from llama_cpp import llama_chat_format
from llama_cpp.llama_chat_format import ChatFormatterResponse

class MarxFormatter(llama_chat_format.ChatFormatter):
    _roles: dict[str, str] = {
        "user": "### HUMAN:",
        "assistant": "### RESPONSE:",
        "system": "### SYSTEM:",
        "function": "### FUNCTION:"
    }

    def marx_message(self, msg: ChatCompletionRequestMessage | ChatCompletionResponseMessage) -> str:
        role = self._roles.get(msg["role"], None)
        if role is None:
            raise Exception("Unknown role")
        return f"""{role}\n{msg["content"]}"""

    def call(self, messages: List[ChatCompletionRequestMessage | ChatCompletionResponseMessage], **kwargs: Any) -> ChatFormatterResponse:
        return ChatFormatterResponse(
            prompt = "\n\n".join(self.marx_message(msg) for msg in messages) + "\n\n" + self._roles["assistant"] + "\n",
            stop = [x for x in self._roles.values()],
            **kwargs
        )
    def __call__(self, messages: List[ChatCompletionRequestMessage | ChatCompletionResponseMessage], **kwargs: Any) -> ChatFormatterResponse:
        return self.call(messages, **kwargs)