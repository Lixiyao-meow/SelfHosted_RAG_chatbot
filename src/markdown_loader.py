import glob
import os

from langchain.document_loaders import UnstructuredMarkdownLoader

class MarkdownLoader():

    def __init__(self, markdown_path):
        self.markdown_path = markdown_path

    def load_batch(self):
        docs = []
        for md_file in glob.glob(os.path.join(self.markdown_path, '*.md')):
            loader = UnstructuredMarkdownLoader(md_file)
            data = loader.load()
            docs.append(data[0])
        return docs