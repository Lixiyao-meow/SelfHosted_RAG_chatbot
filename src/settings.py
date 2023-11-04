from __future__ import annotations
from typing import Optional, Sequence

from atro_args import InputArgs
from pydantic import BaseModel

class Settings(BaseModel):
    HOST: str
    PORT: int
    MARKDOWN_PATH: str = "data"
    EMBED_MODEL_NAME: str = "sentence-transformers/all-mpnet-base-v2"
    MODEL_PATH: str = "model/model.gguf"

def load_settings(cli_args: Optional[Sequence[str]] = None) -> Settings:
    return InputArgs(prefix="RAG").populate_cls(Settings, cli_args)