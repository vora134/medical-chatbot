from flask import Flask , request , render_template
from src.helper import download_embbeding_model
from langchain_pinecone import PineconeVectorStore
from langchain_core.runnables import RunnableLambda,RunnableParallel,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from src.prompt import *
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()


PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY




embeddings =download_embbeding_model()

index_name = "medicalchatbot" 
# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

prompt = PromptTemplate(
    template="""
    Use the following pieces of information to answer the user's question.
    If you don't know the answer, just say that you don't know, didn't try to make up an answer.
    
    Context: {context}
    Question: {question}
    
    Only return the helpful answer below and nothing else.
    Helpful answer:
    """,
    input_variables=["context", "question"]
)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

chatModel = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

main_chain = parallel_chain | prompt | chatModel | parser

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = main_chain.invoke(msg)
    return response

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)

