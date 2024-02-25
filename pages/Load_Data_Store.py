import streamlit as st
from dotenv import load_dotenv
from admin_utils import *
import os

def main():
    load_dotenv()
    st.set_page_config("Dump PDF to Pinecone - Vector Store")
    st.title("Please upload your PDF files. . . 📁")

    pdf = st.file_uploader("Only PDF files allowed", type=["pdf"])

    if pdf is not None:
        with st.spinner("Wait for it. . . "):
            text = read_pdf_data(pdf_file=pdf)
            st.write("✅Reading PDF done")

            #create chunks
            docs_chunks = split_data(text=text)
            #st.write(docs_chunks)
            st.write("✅Splitting data into chunks done")

            #create embeddings
            embeddings = create_embeddings()
            st.write("✅Creating embeddings instance done")

            #Bulid the vecter store (Push the PDF data embeddings)
            pinecone_api_key = os.getenv("PINECONE_API_KEY")
            push_to_pinecone(pinecone_api_key=pinecone_api_key, pinecone_environment="gcp-starter", pinecone_index_name="tickets", embeddings=embeddings, docs_chunk=docs_chunks)
        st.success("Successfully pushed the embeddings to Pinecone")

if __name__ == '__main__':
    main()