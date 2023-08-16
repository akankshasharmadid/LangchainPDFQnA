import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
## Facebook AI similarity Search
from langchain.vectorstores import FAISS
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
#from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text+=page.extract_text()
    return text
    
def get_chunks_text(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len)
    chunk_text = text_splitter.split_text(text)
    return chunk_text


def get_vectorspace(chunks):
    #Hugging Face
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')

    ### ChatGPT OPENAPI
    #embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vector_store


def get_conversation_buffer(vector_space):
    ## Hugging Face
    llm = HuggingFaceHub(
        repo_id = "google/flan-t5-xxl",
        model_kwargs={"temperature":0.5, "max_length": 512}
    )

    ### ChatGPT LLM
    ###llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vector_space.as_retriever(),
        memory = memory
    )
    return conversation_chain

def generate_output(prompt):
    response = st.session_state.conversation({'question': prompt})
    st.session_state.chat_history = response['chat_history']
    for i,res in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(f"User: {res.content}")
        else:
            st.write(f"AI: {res.content}")
        #st.write(f"Assistant:{res['AIMessage']}")


def main():
    #load_dotenv()
    st.set_page_config(layout="wide")
    st.header('Unlocking Answers Across Multiple PDFs: The Langchain Q&A ')
    #st.text_input('Ask any question related to the PDF uploaded')
    prompt = st.chat_input("Ask any question related to the PDF uploaded")
    if prompt:
        generate_output(prompt)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    with st.sidebar:
        text = ''
        st.header('Upload the pdf files')
        pdf_docs = st.file_uploader('Choose the pdf file',accept_multiple_files=True)
        if st.button('Upload'):
            with st.spinner('Working on it..'):
                ## Read pdf text
                raw_text = get_pdf_text(pdf_docs)
                ## Get data in chunks 
                raw_chunks = get_chunks_text(raw_text)
                #st.write(raw_chunks)
                ## Get vector space for the chunks
                embeddings = get_vectorspace(raw_chunks)
                ## conversation buffer
                st.session_state.conversation = get_conversation_buffer(embeddings)

if __name__ == "__main__":
    main()
    
