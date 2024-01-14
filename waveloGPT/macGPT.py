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
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

"""
Mac GPT
"""
from langchain.document_loaders import TextLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.word_document import Docx2txtLoader
from langchain.document_loaders.json_loader import JSONLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

def filter_plans(plans, plan_numbers):
    filtered_plans = [plan for plan in plans if plan["id"] in plan_numbers]
    return json.dumps(filtered_plans, indent=4)

os.environ["OPENAI_API_KEY"] = apikey
image = Image.open("assets/maclogo.png")
image = image.resize((300, 200))
embedding = OpenAIEmbeddings()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
persistence = "./waveloGPT/store"

tab1,tab2 = st.tabs(["Document Reader", "Recommendation"])

with tab1:
    st.header("Document Reader")
    st.image(image, caption="mcmaster university")
    st.title("McMaster GPT")




    # Text Spliter
    splited_text = []
    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=0)

    # File Uploader
    documents = []
    uploaded_files = st.file_uploader("Upload your File", accept_multiple_files=True, key="file_uploader_1")

    for uploaded_file in uploaded_files:
        if uploaded_file:
            target_dir = os.path.join("./waveloGPT/loader", str(datetime.date.today()))
            os.makedirs(target_dir, exist_ok=True)
            target_name = uploaded_file.name
            target_path = os.path.join(target_dir, target_name)

            with open(target_path, "wb") as f:
                f.write(uploaded_file.getvalue())

            if uploaded_file.type.endswith("plain"):
                documents+=TextLoader(file_path=target_path,autodetect_encoding=True).load()
            elif uploaded_file.type.endswith("json"):
                documents+=JSONLoader(
                    file_path=target_path, jq_schema=".", text_content=False
                ).load()
            elif uploaded_file.type.endswith("pdf"):
                documents+=PyPDFLoader(target_path).load()
            elif uploaded_file.type.endswith("wordprocessingml.document"):
                documents+=Docx2txtLoader(target_path).load()


    # if splited_text:
    #     vectordb = Chroma.from_documents(splited_text, embedding)
    #     index = VectorStoreIndexWrapper(vectorstore=vectordb)
    if documents:
        # splited_text += text_splitter.split_documents(documents)
        vectordb = FAISS.from_documents(documents, embedding)
        index = VectorStoreIndexWrapper(vectorstore=vectordb)

    # prompt handling
    prompt = st.text_input("Plug in your prompt here")
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="""
        You are a chatbot answering questions. Answer the question at the end. 
        If you don't know the answer, say that you don't know,
        don't try to make up an answer.

        Question: {question}
        Helpful Answer:
        """,
    )

    if prompt and documents:
        chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model="gpt-3.5-turbo-16k-0613"),
            retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
        )
        result = chain.run(prompt)
        st.write(result)

        with st.expander("Document Similarity Search"):
            # Find the relevant pages
            search = vectordb.similarity_search_with_score(prompt)
            search.sort(key=lambda x: (x[1]))

            st.write(search[0][0].page_content)
            st.write(f"source from:{search[0][0].metadata['source']}")

with tab2:
    st.header("Recommendation")
    st.image(image, caption="mcmaster university")
    st.title("McMaster GPT")
    prompt = st.text_area('Tell us what you are looking for in Internet Plans')

    function_descriptions = [
    {
        "name": "get_internet_plans_ids",
        "description": "Get an array of ids that mostly fit the user's preference and will be used to searched a json file with various internet plans",
        "parameters": {
            "type": "object",
            "properties": {
                "prod_id": {
                    "type": "array",
                    "description": "The array of the IDs of the selected internet plans which match the user's preference",
                    "items": {
                        "type": "string",
                        "description": "the ID of the selected internet plans refer to the given json schema"
                    }
                },
            },
            "required": ["prod_id"],
        },
    }
]

    prompt_template = PromptTemplate.from_template('based on the following user provided internet plan preferences : {preferences} and on the following JSON Schema of Internet Plan Attributes: {internet_plan_schema}, create a lucence search query that best fits the prefferences')
    script_template = PromptTemplate(
    input_variables= ['preferences', 'internet_results_json'],
    template='write a description and some interesting facts about each internet plan returned in the following list: {internet_results_json}. Each paragraph must belong to its own bullet point. Add a final summary after the recommended internet plans for why these choices are a good match for these preferences provided:  {preferences}'
)
    llm=ChatOpenAI(model="gpt-3.5-turbo-16k-0613")
    script_memory = ConversationBufferMemory(input_key='internet_results_json', memory_key='chat_history')
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='internet_recommendations_full', memory=script_memory)
    internet_plan_schema = None

    uploaded_file = st.file_uploader("Upload your File", accept_multiple_files=False, key="file_uploader_2")
    if uploaded_file:
            target_dir = os.path.join("./waveloGPT/loader", str(datetime.date.today()))
            os.makedirs(target_dir, exist_ok=True)
            target_name = uploaded_file.name
            target_path = os.path.join(target_dir, target_name)

            with open(target_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            if uploaded_file.type.endswith("json"):
                internet_plan_schema = json.load(uploaded_file)

    if prompt:
        human_msg = prompt_template.format(preferences=prompt, internet_plan_schema=internet_plan_schema)
        first_response = llm.predict_messages(
        [HumanMessage(content=human_msg)], functions=function_descriptions
    )
        query = json.loads(first_response.additional_kwargs["function_call"]["arguments"])["prod_id"]
        internet_results = filter_plans(internet_plan_schema,query)
        result = script_chain.run(preferences=prompt, internet_results_json=internet_results)
        st.write(result)
        with st.expander('Query History'):
            st.info(script_memory.buffer)


