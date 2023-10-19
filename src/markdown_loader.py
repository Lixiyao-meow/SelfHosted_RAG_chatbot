import os
import fnmatch
from typing import List

from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

class MarkdownLoader():

    def __init__(self, markdown_path: str):
        self.markdown_path = markdown_path

    # find all .md files recursively and save the path into metadata
    def load_batch(self) -> List[Document]:
        docs = []
        
        # walk through all files
        for dirpath, dirs, files in os.walk(self.markdown_path): 
            for filename in fnmatch.filter(files, '*.md'):
                md_filepath = os.path.join(dirpath, filename)
                
                # split one file into multiple files based on headers
                splitted_md_files = self.split_markdown_file(md_filepath)

                for file in splitted_md_files:
                    # add relative location and filename of data to metadata
                    file.metadata["dirpath"] = dirpath
                    file.metadata["filename"] = filename
                    docs.append(file)
                    
        return docs
    
    def split_markdown_file(self, md_filepath: str) -> List[Document]:
        
        md_file = open(md_filepath, "r").read()
        
        # logical split with headers
        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
        ]

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        md_header_splits = markdown_splitter.split_text(md_file)
        
        # text should not exceed 1024 tokens
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1024,
            chunk_overlap = 10,
        )
        splits = text_splitter.split_documents(md_header_splits)

        return splits