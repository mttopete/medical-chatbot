import os

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import RetrievalQA
from langchain.llms import CTransformers
from langchain.prompts import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

from src.helper import download_embeddings
from src.prompt import *

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = "gcp-starter"

embedding = download_embeddings()

Pinecone(api_key = PINECONE_API_KEY,
            envoirment = PINECONE_API_ENV)

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index(index_name,embedding)

prompt = PromptTemplate(template = prompt_template, input_variables = ["context","question"])
chain_type_kwargs={"prompt":prompt}

llm = CTransformers(model = "model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type = "llama",
                    config={'max_new_tokens':512,
                            'temperature':0.8})

qa = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = "stuff",
    retriever = docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents = True,
    chain_type_kwargs=chain_type_kwargs
)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get",methods=["GET","POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query":input})
    print("Response: " ,result["result"])
    return str(result["result"])

if __name__ == "__main__":
    app.run(debug=True)