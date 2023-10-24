import os 
from creds import apikey

import streamlit as st 
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.indexes import VectorstoreIndexCreator
import pytesseract
from PIL import Image



os.environ['OPENAI_API_KEY'] = apikey
image = Image.open('./waveloGPT/assets/maclogo.png')
image = image.resize((600, 400))
st.image(image, caption='mcmaster university')
st.title('McMaster GPT')
persistence = "./waveloGPT/store/"
embedding = OpenAIEmbeddings()
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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








if persistence and os.path.exists(persistence):
    vectordb = Chroma(persist_directory=persistence, embedding_function=embedding)
    index = VectorStoreIndexWrapper(vectorstore=vectordb) 
else:
    text_loader_kwargs={'autodetect_encoding': True}
    loader = DirectoryLoader('./waveloGPT/loader', glob="**/*.txt",loader_cls=TextLoader,loader_kwargs=text_loader_kwargs)

    documents = loader.load()
    if persistence:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"./waveloGPT/store/"}).from_documents(documents)
    else:
        index = VectorstoreIndexCreator().from_documents(documents)
    vectordb = Chroma(persist_directory=persistence, embedding_function=embedding)
    index = VectorStoreIndexWrapper(vectorstore=vectordb) 

 
chain = RetrievalQA.from_chain_type(
  llm=ChatOpenAI(model="gpt-3.5-turbo-16k-0613"),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

#chain.combine_documents_chain.llm_chain.prompt = prompt_template

# prompt = "what do you know about Kafka Multi Tenant notes?"
# prompt = "what do you know about ISOS, and can you walk me through everything you know?"
# prompt = "How do you rebuild ISOS services to AWS ECR?"
#prompt = "How do I provision nomad on EC2"
if prompt:
    st.write(chain.run(prompt))

    with st.expander('Document Similarity Search'):
        # Find the relevant pages
        search = vectordb.similarity_search_with_score(prompt) 
        # Write out the first 
        st.write(search[0][0].page_content)
        st.write(f"source from:{search[0][0].metadata['source']}")
