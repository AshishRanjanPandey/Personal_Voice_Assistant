__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from gtts import gTTS
import base64

st.set_page_config(page_title="LocalVoiceIntel - Gemini Voice Assistant", page_icon="🎙️", layout="wide")

# App Header
st.title("🎙️ LocalVoiceIntel (Gemini Edition)")
st.subheader("Your Free Custom Mini Smart Voice Assistant")
st.markdown("Upload documents, ask questions, and receive spoken responses powered 100% free by Google Gemini AI Studio.")

# Sidebar for Setup & API Key
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if "GOOGLE_API_KEY" in st.secrets:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
        st.success("🤖 Google API Key loaded securely from Secrets!")
    else:
        google_api_key = st.text_input("Enter Google Gemini API Key", type="password", help="Get your free key from Google AI Studio.")
    
    st.markdown("---")
    st.header("📁 Document Knowledge Base")
    uploaded_files = st.file_uploader("Upload local source documents (TXT, PDF)", accept_multiple_files=True, type=["txt", "pdf"])
    
    if uploaded_files:
        os.makedirs("data", exist_ok=True)
        for uploaded_file in uploaded_files:
            with open(os.path.join("data", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getvalue())
        st.success(f"Saved {len(uploaded_files)} files to data folder.")

   if st.button("🔄 Build/Refresh Vector DB", type="primary"):
        if not google_api_key:
            st.error("Please provide a Google API Key first!")
        elif not os.path.exists("data") or len(os.listdir("data")) == 0:
            st.error("The data folder is empty. Please upload documents first.")
        else:
            status_box = st.info("Starting processing... Please wait.")
            try:
                # 🛑 ADD THIS LINE TO KILL ACTIVE BACKGROUND DATABASE CONNECTIONS
                vector_store = None
                
                # Force clean the tmp directory explicitly to prevent read/write conflicts
                if os.path.exists("/tmp/chroma_db"):
                    shutil.rmtree("/tmp/chroma_db")
                os.makedirs("/tmp/chroma_db", exist_ok=True)

# Audio Auto-play helper
def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f'''
            <audio autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            '''
        st.markdown(md, unsafe_allow_html=True)

# Main Application Logic
if not google_api_key:
    st.info("⚠️ Please enter your Google Gemini API Key in the sidebar or save it in Streamlit Secrets to begin.")
else:
    if os.path.exists("/tmp/chroma_db") or "db_built" in st.session_state:
        embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=google_api_key)
        vector_store = Chroma(persist_directory="/tmp/chroma_db", embedding_function=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2, google_api_key=google_api_key)
        
        system_prompt = (
            "You are a friendly, helpful custom voice assistant. Answer the user's question "
            "using only the provided context. If you do not know the answer, say "
            "'I am sorry, I couldn't find that in my local knowledge base.' Keep your response "
            "concise, friendly, and natural for voice synthesis conversion.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        st.subheader("💬 Ask Your Assistant")
        user_query = st.text_input("Type your question or simulate voice command input:")
        
        if user_query:
            with st.spinner("Searching local knowledge base and synthesizing audio reply..."):
                try:
                    response = rag_chain.invoke({"input": user_query})
                    answer = response["answer"]
                    
                    st.markdown("### 🤖 Assistant Response:")
                    st.write(answer)
                    
                    tts = gTTS(text=answer, lang='en')
                    audio_file_path = "response.mp3"
                    tts.save(audio_file_path)
                    
                    st.audio(audio_file_path, format="audio/mp3")
                    autoplay_audio(audio_file_path)
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("📥 Vector Database not found. Please upload documents and click 'Build/Refresh Vector DB' in the sidebar.")
