from langchain.document_loaders import UnstructuredMarkdownLoader

import glob
import os
import dotenv

dotenv.load_dotenv()
markdown_path = os.getenv("MARKDOWN_PATH")

documents = []
for md_file in glob.glob(os.path.join(markdown_path, '*.md')):
    loader = UnstructuredMarkdownLoader(md_file)
    data = loader.load()
    documents.append(data)

print(len(documents))