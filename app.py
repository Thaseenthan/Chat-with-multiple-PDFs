import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template


# ---------- PDF Processing ----------
def get_pdf_chunks_and_metadata(pdf_docs, chunk_size=5000):
    """Extract per-page text, split into chunks, and keep metadata aligned."""
    text_chunks = []
    metadatas = []
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=100,
        length_function=len
    )

    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for i, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if not page_text:
                continue
            # Split page into chunks
            chunks = splitter.split_text(page_text)
            text_chunks.extend(chunks)
            # Duplicate metadata for each chunk
            metadatas.extend([{
                "source": pdf.name,
                "page": i + 1
            }] * len(chunks))

    return text_chunks, metadatas


def get_vectorstore(text_chunks, metadatas):
    embeddings = CohereEmbeddings(
        model="embed-english-light-v3.0",
        cohere_api_key="07pGZKapCMVuBSo1kpw1qX8dNbeZEiWQF4sGiH62"
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings, metadatas=metadatas)


# ---------- Conversation Chain ----------
def get_conversation_chain(vectorstore):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"   # ✅ only save the answer, not the sources
    )
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        return_source_documents=True
    )


def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})

    # Append new user + bot messages to history manually
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.session_state.chat_history.append(("user", user_question))
    st.session_state.chat_history.append(("bot", response["answer"]))

    # Render chat
    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)

    # Show sources
    if "source_documents" in response:
        st.subheader("📖 Sources")
        for doc in response["source_documents"]:
            source = doc.metadata.get("source", "Unknown PDF")
            page = doc.metadata.get("page", "N/A")
            st.markdown(f"- **{source}** (page {page})")

# ---------- Streamlit UI ----------
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []  # ✅ initialize as empty list

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question and st.session_state.conversation:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on process",
            accept_multiple_files=True
        )
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                text_chunks, metadatas = get_pdf_chunks_and_metadata(pdf_docs)
                st.write(f"🔹 Extracted {len(text_chunks)} chunks")

                vectorstore = get_vectorstore(text_chunks, metadatas)
                st.success("✅ Vector store created!")

                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == '__main__':
    main()
