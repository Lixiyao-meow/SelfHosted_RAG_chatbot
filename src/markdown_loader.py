import os
import fnmatch

from langchain.document_loaders import TextLoader

class MarkdownLoader():

    def __init__(self, markdown_path):
        self.markdown_path = markdown_path

    # find all .md files recursively and save the path
    def load_batch(self):
        docs = []
        for dirpath, dirs, files in os.walk(self.markdown_path): 
            for filename in fnmatch.filter(files, '*.md'):
                md_file = os.path.join(dirpath, filename)
                loader = TextLoader(md_file, encoding="utf-8")
                data = loader.load()

                # add relative location and filename of data to metadata
                data[0].metadata["dirpath"] = dirpath
                data[0].metadata["filename"] = filename
                docs.append(data[0])

        return docs