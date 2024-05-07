import streamlit as st
from streamlit_modal import Modal

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def display_how_to():
    model = Modal(key = "how_to", title = "🚀 How to Use The App")

    with model.container():
        st.text("▶️ Select Menu.")
        st.text("▶️ Insert your Open AI API key.")
        st.text("▶️ Upload your PDF document.")
        st.text("▶️ Choose the menu.")
        st.text("▶️ Summarizer: provide the general summary of the PDF document.")
        st.text("▶️ Q&A: ask over the PDF file.")


