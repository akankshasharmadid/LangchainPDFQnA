import streamlit as st
import PyPDF2

def get_pdf_text(pdf_docs):
    pdfdoc = PyPDF2.PdfReader(pdf_docs)
    pdfdoc.documentInfo
    


def main():
    st.title('Langchain Question and Answers: Multiple PDFs')
    st.text_input('Ask any question related to the PDF uploaded')
    with st.sidebar:
        text = ''
        st.header('Upload the pdf files')
        pdf_docs = st.file_uploader('Choose the pdf file',accept_multiple_files=True)
        st.button('Upload')
        ## Read pdf text
        get_pdf_text(pdf_docs)

if __name__ == "__main__":
    main()
    