import streamlit as st
from dotenv import  load_dotenv
from user_utils import *
import os

if "HR_tickets" not in st.session_state:
    st.session_state["HR_tickets"] = []
if "IT_tickets" not in st.session_state:
    st.session_state["IT_tickets"] = []
if "Transport_tickets" not in st.session_state:
    st.session_state["Transport_tickets"] = []

def main():
    load_dotenv()
    st.header("Automatic Ticket Classification Tool")

    st.write("We are here to help you, please ask your question:")
    user_input = st.text_input("🔍")

    if user_input:
        #create embeddings instance
        embeddings = create_embeddings()
        #Function to pull index data from Pinecone
        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        index = pull_from_pinecone(pinecone_api_key=pinecone_api_key, pinecone_environment="gcp-starter", pinecone_index_name="tickets", embeddings=embeddings)
        #This function will help us in fetching the top relevent documents from our vector store - Pinecone Index
        relevant_docs = get_similar_docs(index=index, query=user_input, k=2)
        #This will return the fine tuned response by LLM
        response = get_answer(docs=relevant_docs, user_input=user_input)
        st.write(response)

        #Button to create a ticket with respective department
        button = st.button("Submit ticket?")

        if button:
            st.write("rasie ticket")

            embeddings = create_embeddings()
            query_result = embeddings.embed_query(user_input)

            #Load ML model to predict department the complaint belongs to
            department_value = predict(query_result)
            st.write(f"Your ticket has been submit to {department_value}.")

            if department_value == "HR":
                st.session_state["HR_tickets"].append(user_input)
            elif department_value =="IT":
                st.session_state["IT_tickets"].append(user_input)
            else:
                st.session_state["Transport_tickets"].append(user_input)

if __name__ == '__main__':
    main()