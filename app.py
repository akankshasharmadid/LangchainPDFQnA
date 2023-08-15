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
    
    embeddings = HuggingFaceInstructEmbeddings(model_name = 'hkunlp/instructor-xl')
    vector_store = FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vector_store


def get_conversation_buffer(vector_space):
    llm = HuggingFaceHub(
        repo_id = "google/flan-t5-xxl",
        model_kwargs={"temperature":0.5, "max_length": 512}
    )
    memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
    conversation_chain = ConversationalRetrievalChain(
        llm = llm,
        retriever = vector_space.as_retriever(),
        memory = memory
    )
    return conversation_chain

def generate_output(prompt):
    st.write(f"User: {prompt}")


def main():
    st.title('Langchain Question and Answers: Multiple PDFs')
    #st.text_input('Ask any question related to the PDF uploaded')
    prompt = st.chat_input("Ask any question related to the PDF uploaded")
    if prompt:
        generate_output(prompt)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
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
    