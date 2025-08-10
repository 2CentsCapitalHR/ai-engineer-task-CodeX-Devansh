import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

DOCUMENTS_PATH = "adgm_documents"
INDEX_STORE_PATH = "adgm_faiss_index_local"

def create_vector_store():
    print("Loading documents...")
    pdf_loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True)
    word_loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, show_progress=True)
    documents = pdf_loader.load() + word_loader.load()
    
    if not documents:
        print(f"ERROR: No documents found in '{DOCUMENTS_PATH}'.")
        return

    print(f"Splitting {len(documents)} documents into very small chunks (size 250)...")
    text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)
    
    print(f"Split into {len(docs)} chunks. Initializing local embeddings model...")
    embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
    
    print("Creating and saving local FAISS vector store...")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(INDEX_STORE_PATH)
    print(f"SUCCESS: Local vector store created at '{INDEX_STORE_PATH}'")

if __name__ == "__main__":
    create_vector_store()