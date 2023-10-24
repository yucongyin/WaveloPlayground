import os 
import datetime
import pytesseract
import streamlit as st 
import json

from creds import apikey
from PIL import Image

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.indexes import VectorstoreIndexCreator
from langchain.docstore.document import Document

"""
File Loaders
"""
from langchain.document_loaders import TextLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain.document_loaders.json_loader import JSONLoader



os.environ['OPENAI_API_KEY'] = apikey
image = Image.open('./waveloGPT/assets/maclogo.png')
image = image.resize((600, 400))

st.image(image, caption='mcmaster university')
st.title('McMaster GPT')
persistence = "./waveloGPT/store"
embedding = OpenAIEmbeddings()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


vectordb = Chroma(persist_directory=persistence, embedding_function=embedding)

# File uploader
up_file = st.file_uploader("Upload your File")
if up_file:
    target_dir = os.path.join("./waveloGPT/loader", str(datetime.date.today()))
    os.makedirs(target_dir, exist_ok=True)
    target_name = up_file.name
    target_path = os.path.join(target_dir, target_name)
    
    with open(target_path, 'wb') as f:
        f.write(up_file.getvalue())

    documents = None
    if up_file.type.endswith("plain"):
        documents = TextLoader(target_path).load()
    elif up_file.type.endswith("json"):
        documents = JSONLoader(
            file_path=target_path,
            jq_schema='.',
            text_content=False
        ).load()
    elif up_file.type.endswith("pdf"):
        documents = PyPDFLoader(target_path).load()
    elif up_file.type.endswith("wordprocessingml.document"):
        documents = Docx2txtLoader(target_path).load()

    if documents:
        index = VectorstoreIndexCreator().from_documents(documents)
    else:
        raise ValueError("The file is not supported.")


#prompt handling
prompt = st.text_input('Plug in your prompt here') 
prompt_template = PromptTemplate(
    input_variables = ["question"], 
    template= """
    You are a chatbot answering questions. Answer the question at the end. 
    If you don't know the answer, say that you don't know,
    don't try to make up an answer.

    Question: {question}
    Helpful Answer:
    """
)
 
#chain.combine_documents_chain.llm_chain.prompt = prompt_template
# prompt = "what do you know about Kafka Multi Tenant notes?"
# prompt = "what do you know about ISOS, and can you walk me through everything you know?"
# prompt = "How do you rebuild ISOS services to AWS ECR?"
#prompt = "How do I provision nomad on EC2"

if up_file and prompt:
    chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-3.5-turbo-16k-0613"),
        retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        )
    
    st.write(chain.run(prompt))

    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = vectordb.similarity_search_with_score(prompt) 
        print(search)
        # Write out the first 
        # st.write(search[0][0].page_content)
        # st.write(f"source from:{search[0][0].metadata['source']}")
