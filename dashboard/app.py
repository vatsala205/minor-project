import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from models.random_forest import train_random_forest
from models.bert_model import train_bert
from models.rag_model import RiskRAG
from utils.preprocessing import load_data

st.set_page_config(layout="wide")

st.title("Behavioral Risk Stratification System")

# Load data
df = load_data()
df = df.dropna(subset=["Risk_Level"])

st.subheader("Dataset Overview")
st.write("Total Samples:", len(df))

st.subheader("Risk Distribution")
risk_counts = df["Risk_Level"].value_counts()
st.dataframe(risk_counts)

fig1, ax1 = plt.subplots()
risk_counts.plot(kind="bar", ax=ax1)
ax1.set_title("Risk Level Distribution")
st.pyplot(fig1)

st.divider()

# Random Forest Section
st.header("Random Forest Model")

if st.button("Train Random Forest"):
    results = train_random_forest()

    st.success("Random Forest Training Complete")

    st.write("Accuracy:", round(results["accuracy"], 4))

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(results["report"]).transpose())

    st.subheader("Top 10 Feature Importances")
    fig2, ax2 = plt.subplots()
    results["feature_importances"].head(10).plot(kind="bar", ax=ax2)
    st.pyplot(fig2)

st.divider()

# BERT Section
st.header("DistilBERT Model")

if st.button("Train DistilBERT"):
    results = train_bert()

    st.success("DistilBERT Training Complete")

    st.write("Accuracy:", round(results["accuracy"], 4))

    st.subheader("Classification Report")
    st.dataframe(pd.DataFrame(results["report"]).transpose())

st.divider()

st.header("RAG Explanation Module")

if st.button("Initialize RAG Engine"):
    st.session_state.rag = RiskRAG()
    st.success("RAG Engine Ready")

if "rag" in st.session_state:
    sample_index = st.number_input(
        "Select Individual Index (0 to {})".format(len(df)-1),
        min_value=0,
        max_value=len(df)-1,
        value=0
    )

    selected_row = df.iloc[sample_index]
    query_text = st.session_state.rag.convert_row_to_text(selected_row)

    if st.button("Generate Explanation"):
        explanation, similar_cases = st.session_state.rag.generate_explanation(query_text)

        st.subheader("Generated Explanation")
        st.write(explanation)

        st.subheader("Top Similar Cases")
        st.dataframe(similar_cases[["Risk_Level"]])