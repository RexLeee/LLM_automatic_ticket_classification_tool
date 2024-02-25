import streamlit as st
from admin_utils import *
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib

if "clean_data" not in st.session_state:
    st.session_state["clean_data"] = ""
if "sentences_train" not in st.session_state:
    st.session_state["sentences_train"] = ""
if "sentences_test" not in st.session_state:
    st.session_state["sentences_test"] = ""
if "labels_train" not in st.session_state:
    st.session_state["labels_train"] = ""
if "labels_test" not in st.session_state:
    st.session_state["labels_train"] = ""
if "svm_classifier" not in st.session_state:
    st.session_state["svm_classifier"] = ""

st.title("Let's build our Model. . . ")

tab_titles = ["Data Preprocessing", "Model Training", "Model Evaluation", "Save Model"]
tabs = st.tabs(tab_titles)

#Data Processing Tab
with tabs[0]:
    st.header("Data Processing")
    st.write("Here we preprocess the data. . . ")

    data = st.file_uploader("Uplaod CSV File", type="csv")

    button = st.button("Load data", key="data")

    if button:
        with st.spinner("Wait for it. . . "):
            our_data = read_data(data)
            embeddings = get_embeddings()
            st.session_state["clean_data"] = create_embeddings(our_data, embeddings)
        st.success("Data loaded successcully!")

#Model Training Tab
with tabs[1]:
    st.header("Model Training")
    st.write("Here we train the model. . . ")

    button = st.button("Train model", key="train")

    if button:
        with st.spinner("Wait for it. . . "):
            st.session_state["sentences_train"], st.session_state["sentences_test"], st.session_state["labels_train"], st.session_state["labels_test"] = split_train_test_data(st.session_state["clean_data"])
            st.session_state["svm_classifier"] = make_pipeline(StandardScaler(), SVC(class_weight="balanced"))
            st.session_state["svm_classifier"].fit(st.session_state["sentences_train"], st.session_state["labels_train"])
        st.success("Model trained successcully!")
    
#Model Evaluation Tab
with tabs[2]:
    st.header("Model Evaluation")
    st.write("Here we evaluate the model. . . ")

    button = st.button("Evaluate model", key="evaluate")

    if button:
        with st.spinner("Wait for it. . . "):
            accuracy_score = get_score(st.session_state["svm_classifier"], st.session_state["sentences_test"], st.session_state["labels_test"])
            st.success(f"Validation accuracy is {accuracy_score*100}%!")

            st.write("A sample run:")

            text = "Rude driver with scary driving"
            st.write("***Our issue*** : " + text)
            embeddings = get_embeddings()
            query_result = embeddings.embed_query(text)

            result = st.session_state["svm_classifier"].predict([query_result])

            st.write(f"Department it belongs to: {result[0]}")

        st.success("Model evaluated successcully!")

#Save Model Tab
with tabs[3]:
    st.header("Save Model")
    st.write("Here we save the model. . . ")

    button = st.button("save model", key="save")

    if button:
        with st.spinner("Wait for it. . . "):
            joblib.dump(st.session_state["svm_classifier"], 'modelsvm.pk1')
        st.success("Model saved successcully!")