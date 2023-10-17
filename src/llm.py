import os
import dotenv

from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# initialize the model

# TODO: change model cause current one is not good
dotenv.load_dotenv()
model_path = os.getenv("MODEL_PATH")

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=0, # TODO: make it work with GPU
    #n_batch=512,
    temperature=0.75,
    max_tokens=200,
    callback_manager=callback_manager,
    verbose=True, # Verbose is required to pass to the callback manager
)

llm("What is the capital of Hungary?")