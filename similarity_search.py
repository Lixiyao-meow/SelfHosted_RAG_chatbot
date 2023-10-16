import os
import dotenv
import markdown_loader

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant

dotenv.load_dotenv()
markdown_path = os.getenv("MARKDOWN_PATH")

# load markdown files
loader = markdown_loader.MarkdownLoader(markdown_path)
docs = loader.load_batch()

# load embedding model
# TODO: Is it possible to hold it locally?
model_path = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(model_name=model_path,
                                   model_kwargs=model_kwargs,
                                   encode_kwargs=encode_kwargs)

# create vector database
vectorstore = Qdrant.from_documents(
    docs,
    embeddings,
    location=":memory:",  # Local mode with in-memory storage only
    collection_name="MD_files",
)

# query the vector database
query = "How to get the nodes running kubernetes?"

answer = vectorstore.similarity_search(query, k=1)
print(answer)