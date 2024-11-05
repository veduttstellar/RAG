# from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
# from langchain.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil
import glob

from pathlib import Path


import nltk

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

CHROMA_PATH = "chroma"
DATA_PATH = "data/books/"


def main():
    print("hello")
    generate_data_store()


def generate_data_store():
    print("generating data store")
    documents = load_documents()
    chunks = split_text(documents)
    print("chunks")
    print(chunks)
    save_to_chroma(chunks)


def load_documents():

    documents = []
    try:
        #documents = loader.load()
        for md_file in Path(DATA_PATH).glob("*.md"):
            content = md_file.read_text(encoding='utf-8')
            document = Document(page_content=content, metadata={"source": str(md_file)})
            documents.append(document)
            #print(f"Loaded content from {md_file}")

        return documents
 
    except Exception as e:
        print(f"Error loading documents: {e}")
        return []




def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()


if __name__ == "__main__":
    main()
