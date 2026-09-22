import streamlit as st
import requests

st.title("Text Analyzer")
text = st.text_area("Enter your text")

if st.button("Analyze"):
    response = requests.post("http://127.0.0.1:8000/analyze", json={"text": text})
    st.json(response.json())
