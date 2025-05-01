from flask import Flask, request, jsonify, render_template
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders import PyMuPDFLoader
import os

app = Flask(__name__)
db = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global db
    pdf = request.files['pdf']
    if pdf:
        pdf.save("temp.pdf")
        loader = PyMuPDFLoader("temp.pdf")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()
        db = FAISS.from_documents(texts, embeddings)
        return jsonify({"message": "PDF uploaded and processed successfully."})
    return jsonify({"error": "No file uploaded."}), 400

@app.route('/ask', methods=['POST'])
def ask():
    global db
    if db is None:
        return jsonify({"error": "No PDF has been uploaded yet."}), 400

    query = request.json.get("question")
    if query:
        docs = db.similarity_search(query)
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        answer = chain.run(input_documents=docs, question=query)
        return jsonify({"answer": answer})
    return jsonify({"error": "No question provided."}), 400

if __name__ == '__main__':
    app.run(debug=True)