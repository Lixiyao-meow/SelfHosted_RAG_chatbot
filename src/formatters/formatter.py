from enum import Enum
from typing import Any, Callable, Iterable, List

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_file_as, parse_yaml_raw_as, to_yaml_str

from llama_cpp import ChatCompletionMessage, ChatCompletionRequestMessage, ChatCompletionResponseMessage
from llama_cpp import llama_chat_format
from llama_cpp.llama_chat_format import ChatFormatterResponse

class RoleModel(BaseModel):
    prefix: str
    suffix: str

class FormatterModel(BaseModel):
    roles: dict[str, RoleModel]

class KnownFormats(Enum):
    Marx = "./format/marx.yaml"
    ChatML = "./format/chatml.yaml"

class GeneralFormatter(llama_chat_format.ChatFormatter):
    def __init__(self, format_path: str) -> None:
        self.model: FormatterModel = parse_yaml_file_as(FormatterModel, format_path)
    
    def stop_keys(self):
        for role in self.model.roles.values():
            yield role.prefix
            if role.suffix is not None and role.suffix.strip() != "":
                yield role.suffix
    
    def format_message(self, message: ChatCompletionRequestMessage) -> str:
        role = self.model.roles.get(message['role'], None)
        if role is None:
            raise Exception(f"Selected format does not support role {message['role']}")
        if type(message["content"]) is not str:
            raise Exception(f"DEVELOPER ERROR - Unhandled message content type {type(message['content'])}:\n{message['content']}")
        return role.prefix + "\n" + message["content"] + role.suffix
        


    def __call__(self, *, messages: List[ChatCompletionRequestMessage], **kwargs: Any) -> ChatFormatterResponse:
        prompt = "\n".join(self.format_message(msg) for msg in messages)
        assistant = self.model.roles.get("assistant")
        if assistant is None:
            raise Exception("Panic - No assistant message type handled")
        prompt += "\n" + assistant.prefix + "\n"
        
        return ChatFormatterResponse(prompt, stop=[k for k in self.stop_keys()])
