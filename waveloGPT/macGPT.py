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
st.write("""<div><svg width="105" id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 717.91 228.78"><defs><style>.cls-1{fill:#000;}</style></defs><path class="cls-1" d="M226.53,91.45a7.36,7.36,0,0,1,6.84,5L249,145.76l15.67-49.29a7.34,7.34,0,0,1,6.7-5h3.8a7.38,7.38,0,0,1,6.85,5l15.51,49.29L313.4,96.47a7.36,7.36,0,0,1,6.84-5h5.33c3,0,4.71,2.13,3.8,5l-23.28,69.07a7.36,7.36,0,0,1-6.84,5H295.6a7.34,7.34,0,0,1-6.7-5l-15.51-47.47-15.52,47.47a7.37,7.37,0,0,1-6.85,5h-3.65a7.36,7.36,0,0,1-6.84-5L217.4,96.47c-.91-2.89.61-5,3.65-5Z"/><path class="cls-1" d="M335.45,131c0-23.74,16.59-41.53,38.19-41.53A37.07,37.07,0,0,1,400.72,101V96.78a5.08,5.08,0,0,1,5.17-5.33h5.32a5.2,5.2,0,0,1,5.33,5.33v68.46a5.2,5.2,0,0,1-5.33,5.32h-5.32a5.07,5.07,0,0,1-5.17-5.32V161a37.07,37.07,0,0,1-27.08,11.56C352,172.54,335.45,154.74,335.45,131Zm16.43,0c0,14.6,10.65,25.71,24.19,25.71s24.34-11.11,24.34-25.71-10.65-25.71-24.34-25.71S351.88,116.4,351.88,131Z"/><path class="cls-1" d="M437.23,91.45a8,8,0,0,1,7.15,4.87l22.36,53.85,22.52-53.85a8,8,0,0,1,7.15-4.87h5.93c3,0,4.41,2.13,3.19,4.87l-29.36,69.37a8,8,0,0,1-7.15,4.87h-4.41a8,8,0,0,1-7.15-4.87L428.1,96.32c-1.22-2.74.15-4.87,3.19-4.87Z"/><path class="cls-1" d="M526.53,136.79c2.43,12.17,12.78,21,26.16,21a26.47,26.47,0,0,0,18.72-7.45,6.81,6.81,0,0,1,4.25-1.68,3.19,3.19,0,0,1,1.22.16l7.3,2.28c2.89.91,3.81,3.65,2,5.93-7.6,9.59-19.47,15.52-33.47,15.52-24.19,0-42.44-17.8-42.44-41.53s17.8-41.53,41.23-41.53c24.49,0,41.22,18.86,41.07,42a5.46,5.46,0,0,1-5.63,5.33ZM551,104.23c-11.71,0-21.45,8.22-24.34,19.93h49.59A25.64,25.64,0,0,0,551,104.23Z"/><path class="cls-1" d="M606.55,61.79a5.2,5.2,0,0,1,5.32-5.33h5.33a5.16,5.16,0,0,1,5.17,5.33V165.24a5.16,5.16,0,0,1-5.17,5.32h-5.33a5.19,5.19,0,0,1-5.32-5.32Z"/><path class="cls-1" d="M676.22,89.48c23.73,0,41.69,17.79,41.69,41.53s-18,41.53-41.69,41.53-41.53-17.8-41.53-41.53S652.49,89.48,676.22,89.48Zm0,67.24c14.3,0,25.25-11.11,25.25-25.71s-11-25.71-25.25-25.71-25.1,11.1-25.1,25.71S662.07,156.72,676.22,156.72Z"/><path class="cls-1" d="M173.74,0h0c-26.86,0-43.05,56.85-60.18,117-6,21-12.09,42.46-18.44,60.2a26.38,26.38,0,0,1-2.45-2.5c-12.52-14.5-15.25-46.39-8.11-94.79.63-4.26,1.46-7.37.69-9.44a4.67,4.67,0,0,0-3.09-3.21,5.07,5.07,0,0,0-5.56,2.36,28.66,28.66,0,0,0-2,3.91C74,75,73.15,76.94,72.14,79.44c-4.09,10-10.94,26.8-19.13,41.37-1.44,2.57-2.82,4.87-4.14,6.94-2.3-31.8-7.09-57.81-26.4-57.81C15,69.94,8.9,73.59,5,80.48-4,96.13.21,124.93,9.62,141c5.59,9.54,12.66,14.58,20.45,14.58h.17a18.81,18.81,0,0,0,10.07-3.33c1,17.6,2.06,35.5,5.13,49.16,1.52,6.76,6.16,27.36,23,27.37h0c12.76,0,22.81-14.62,32-37.24a39.41,39.41,0,0,0,12.34,1.86c9.84,0,20.25-6.16,31-18.29,9-10.17,18.06-24.54,26.3-41.55,16.34-33.71,26.89-72.28,26.89-98.27C196.88,4.58,182.38,0,173.74,0ZM30.15,145.58h-.07c-5.38,0-9.72-6-11.83-9.63-8.72-14.89-11-39.43-4.63-50.49,2.16-3.76,5-5.52,8.85-5.52C35,79.94,38,111.16,39.63,139.88,34.86,144.75,31.69,145.57,30.15,145.58Zm60.4,43.6c-10.06,24.46-17.59,29.6-22.13,29.6h0c-14.5,0-16.61-38.95-18.3-70.25l-.27-5.08a117.33,117.33,0,0,0,11.88-17.74c3.5-6.22,6.73-12.78,9.63-19.06-1.9,25-1.1,57.42,13.74,74.61A35.75,35.75,0,0,0,91.42,187C91.13,187.76,90.84,188.48,90.55,189.18Zm70.45-60c-16.2,33.43-34.69,54.2-48.26,54.2a30,30,0,0,1-8.76-1.21c6.52-17.9,12.68-39.55,19.2-62.43C137.1,70.86,154.43,10,173.74,10c8.72,0,13.14,8.51,13.14,25.29C186.88,59.88,176.72,96.74,161,129.2Z"/></svg></div>""", unsafe_allow_html=True)
st.title('Wavelo GPT')
embedding = OpenAIEmbeddings()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
persistence = "./waveloGPT/store"

tab1,tab2 = st.tabs(["Document Reader", "Recommendation"])

with tab1:
    st.header("Document Reader")
    st.title("Wavelo GPT")




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
    st.title("Wavelo GPT")
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


