# Environment variables:
```env
MARKDOWN_PATH=<path to markdown documents to recursively load>
MODEL_PATH=<llamacpp model path if using LLamaCPP>
USE_INMEMORY_LLM=<true/false for running LLM model inmemory -> LlamaCPP>
EMBED_MODEL_NAME=<name>
EMBEDDING_API=<url for embedding eg.: localhost:8000/v1>
HOST=0.0.0.0
PORT=8000
```

# Development
To create `requirements.txt` use:
```sh
poetry export --without-hashes -f requirements.txt -o requirements.txt
```
OR
```sh
poetry export --without-hashes --without dev -f requirements.txt -o requirements.txt
```

## Local Llama
GPU support for `llama-cpp-python` requires manual installation:
```sh
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python==0.2.11 --force-reinstall --no-cache-dir
```

## Pytorch with CUDA
```sh
pip install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121        
```

This will create a clean requirements.txt installation that pip can use

# Image Building
To build docker image to harbor:
```sh
docker buildx create --use
```
```sh
docker buildx build \
    --tag harbor.atro.xyz/llm/petur:0.0.1 \
    --tag harbor.atro.xyz/llm/petur:latest \
    --file Dockerfile \
    --platform linux/amd64,linux/arm64 \
    --push .
```