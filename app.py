# app.py
import streamlit as st
import os
from bot_core import build_chain

UPLOAD_FOLDER = "uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

st.set_page_config(page_title="AI Chatbot with PDF Memory")
st.title("📚 AI Chatbot with File Upload + Memory")

# File upload
uploaded = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded:
    path = os.path.join(UPLOAD_FOLDER, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())
    st.success(f"Uploaded: {uploaded.name}")

# Show current files
st.subheader("📂 Files in Context")
files = os.listdir(UPLOAD_FOLDER)
for fname in files:
    col1, col2 = st.columns([4, 1])
    col1.write(fname)
    if col2.button("❌", key=f"delete_{fname}"):
        os.remove(os.path.join(UPLOAD_FOLDER, fname))
        st.experimental_rerun()

# Initialize bot + retriever
chain, retriever = build_chain()
session_id = st.session_state.get("session_id", "default")

# Chat interface
st.subheader("💬 Chat")
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

user_input = st.text_input("Your message:")
if st.button("Send") and user_input:
    context_docs = retriever.get_relevant_documents(user_input)
    context = "\n\n".join(doc.page_content for doc in context_docs[:3])
    response = chain.invoke(
        {"input": user_input, "context": context},
        config={"configurable": {"session_id": session_id}}
    )
    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", response.content))

# Display chat
for speaker, message in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {message}")
