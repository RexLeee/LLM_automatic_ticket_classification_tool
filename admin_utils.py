from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.llms.openai import OpenAI
from pinecone import Pinecone as PineconeClient
from langchain_community.vectorstores.pinecone import Pinecone
import pandas as pd
from sklearn.model_selection import train_test_split

def read_pdf_data(pdf_file):
    pdf_page = PdfReader(pdf_file)
    text = ""

    for page in pdf_page.pages:
        text += page.extract_text()
    return text

def split_data(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
    docs = splitter.split_text(text)
    docs_chunks = splitter.create_documents(docs)

    return docs_chunks

def create_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

def push_to_pinecone(pinecone_api_key, pinecone_environment, pinecone_index_name, embeddings, docs_chunk):
    PineconeClient(api_key=pinecone_api_key, environment = pinecone_environment)

    index_name = pinecone_index_name

    index = Pinecone.from_documents(docs_chunk, embeddings, index_name=index_name)
    return index

#----------functions for ML model----------

#read dataset
def read_data(data):
    df = pd.read_csv(data, delimiter=',', header=None)
    return df
#create embeddings instance
def get_embeddings():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embeddings

#Generating embeddings for our input dataset
def create_embeddings(df, embeddings):
    df[2] = df[0].apply(lambda x: embeddings.embed_query(x))
    return df

#Split the data into train set and test set
def split_train_test_data(df):
    sentences_train, sentences_test, labels_train, labels_test = train_test_split(
        list(df[2]), list(df[1]), test_size=0.25, random_state=0
    )
    return sentences_train, sentences_test, labels_train, labels_test

#Get the accuracy of our model
def get_score(svm_classifier, sentences_test, labels_test):
    score = svm_classifier.score(sentences_test, labels_test)
    return score






