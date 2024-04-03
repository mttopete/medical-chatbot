import os

from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.helper import download_embeddings, load_pdf, text_splitter

load_dotenv()

embedding = download_embeddings()
extracted_data = load_pdf("../data/")
text_chunks = text_splitter(extracted_data)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = "gcp-starter"

Pinecone(api_key = PINECONE_API_KEY,
            envoirment = PINECONE_API_ENV)

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_texts([t.page_content for t in text_chunks],embedding,index_name = index_name)