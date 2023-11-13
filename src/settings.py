from __future__ import annotations
from typing import Literal, Optional, Sequence, TypeVar, Union

from atro_args import InputArgs
from pydantic import BaseModel

__PREFIX = "RAG"

class Settings(BaseModel):
    """
    Overall settings for the RAG server
    """
    HOST: str
    PORT: int
    MARKDOWN_PATH: str = "data"
    HOSTED_EMBEDDING: bool = False
    # EMBED_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2" # only required if inmemory embedding is used
    HOSTED_LLM: bool = False
    # MODEL_PATH: str = "model/model.gguf" # only required if inmemory LLM is used
    DATABASE_URL: str = "http://localhost"

class InMemoryEmbeddingSettings(BaseModel):
    EMBED_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    EMBED_DEVICE: Literal["cuda"] | Literal["cpu"] = "cpu" # set to CPU by default to run it easily

class HostedEmbeddingSettings(BaseModel):
    EMBED_API_BASE: str = "http://127.0.0.1"

class InMemoryLlmSettings(BaseModel):
    """
    Settings required for inmemory LLM
    """
    MODEL_PATH: str
    N_GPU_LAYERS: int = 0
    N_BATCH: int = 512
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 2 ** 10 # 1024
    N_CTX: int = 2 ** 10 # 1024
    EMBEDDING: bool = True
    VERBOSE: bool = False

class HostedLlmSettings(BaseModel):
    """
    Settings required for to call hosted LLM
    """
    LLM_API_BASE: str = "http://127.0.0.1/v1"
    LLM_API_KEY: str = "password"

T = TypeVar("T", bound=BaseModel)
def load_settings(cls: type[T], cli_args: Optional[Sequence[str]] = None) -> T:
    return InputArgs(prefix=__PREFIX).populate_cls(cls, cli_args)