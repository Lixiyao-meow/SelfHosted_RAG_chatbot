# Environment variables:
```env
# Settings for hosting and document loading
RAG_MARKDOWN_PATH=<path to markdown documents>
RAG_HOST=0.0.0.0
RAG_PORT=8000

# Settings for Generative Model
RAG_HOSTED_LLM=false
RAG_MODEL_PATH=./models/model.gguf
RAG_N_GPU_LAYERS=32
RAG_VERBOSE=true

# Settings for embedding model
RAG_EMBED_MODEL_NAME=sentence-transformers/all-mpnet-base-v2
RAG_EMBED_DEVICE=cuda
```

# Development

## Install requirements

To create `requirements.txt` use:
```sh
poetry export --without-hashes -f requirements.txt -o requirements.txt
```
OR
```sh
poetry export --without-hashes --without dev -f requirements.txt -o requirements.txt
```

## Install Pytorch with CUDA (required for inmemory embedding model)
```sh
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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