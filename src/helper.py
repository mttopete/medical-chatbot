from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(data):
    loader = DirectoryLoader(data,
                    glob ="*.pdf",
                    loader_cls=PyPDFLoader
                    )
    documents = loader.load()
    return documents

def text_splitter(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,chunk_overlap = 20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def download_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    return embeddings