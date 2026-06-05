from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch
import os

if torch.cuda.is_available():
    active_device = "cuda"
else:
    active_device = "cpu"


def index_documents(pdf_paths: list) -> int:
    """Takes the list of all the path of the pdf and indexes each of them into the ChromDB collection"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={"device": active_device},
    )
    all_chunks = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        document = loader.load()
        chunks = splitter.split_documents(documents=document)
        print(f"Split into {len(chunks)} chunks")
        for chunk in chunks:
            chunk.metadata["source"] = os.path.basename(path)
            chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1
            all_chunks.append(chunk)
    print(f"Created {len(all_chunks)} chunks from {len(pdf_paths)} documents")
    vectorstore = Chroma(
        persist_directory="./chroma_store",
        embedding_function=embedding_model,
        collection_name="doc-buddy",
    )
    vectorstore.add_documents(documents=all_chunks)
    print(f"Indexed {vectorstore._collection.count()} chunks into ChromaDB")
    return len(all_chunks)
