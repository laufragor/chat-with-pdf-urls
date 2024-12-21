import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import requests
import io
import re

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def is_valid_pdf_url(url):
    """Validate if the URL is https (secure) and points to a PDF file"""
    pdf_pattern = r'^https://.*\.pdf$'
    return bool(re.match(pdf_pattern, url.strip(), re.IGNORECASE))


def download_pdf_from_url(url):
    """Download PDF from URL and return as a BytesIO object"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        return io.BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading PDF from {url}: {str(e)}")
        return None


def get_pdf_text(urls):
    text = ""
    for url in urls:
        if is_valid_pdf_url(url):
            pdf_content = download_pdf_from_url(url)
            if pdf_content:
                pdf_reader = PdfReader(pdf_content)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        else:
            st.warning(f"Invalid PDF URL: {url}")
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    print(response)
    st.write(response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini 💬")

    # URL input section
    st.subheader("PDF URLs")
    url_input = st.text_area("Enter PDF URLs (one per line)")
    pdf_urls = [url.strip() for url in url_input.split("\n") if url.strip()] if url_input else []
    
    # Process button
    if st.button("Submit & Process"):
        if not pdf_urls:
            st.error("Please provide at least one PDF URL")
            return
        
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_urls)
            if raw_text.strip():  # Check if any text was extracted
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Ready to chat!")
            else:
                st.error("Error: no text could be extracted from the provided PDFs. Please check the URLs and try again.")

    # User question section 
    st.subheader("Ask questions")
    user_question = st.text_input("Ask a question from the PDF Files, then press Enter:")
    if user_question:
        user_input(user_question)


if __name__ == "__main__":
    main()