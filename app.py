import streamlit as st
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from gtts import gTTS
import base64

st.set_page_config(page_title="LocalVoiceIntel - Gemini Voice Assistant", page_icon="🎙️", layout="wide")

# App Header
st.title("🎙️ LocalVoiceIntel (Gemini Edition)")
st.subheader("Your Free Custom Mini Smart Voice Assistant")
st.markdown("Upload documents, ask questions, and receive spoken responses powered 100% free by Google Gemini AI Studio.")

# Initialize an empty in-memory vector store in session state if it doesn't exist
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None

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
            status_box = st.info("Processing document chunks entirely in memory...")
            try:
                # Load text files and PDFs safely with clean encoding mappings
                txt_loader = DirectoryLoader("data", glob="*.txt", loader_cls=lambda p: TextLoader(p, encoding="utf-8"))
                pdf_loader = DirectoryLoader("data", glob="*.pdf", loader_cls=PyPDFLoader)
                
                docs = []
                docs.extend(txt_loader.load())
                docs.extend(pdf_loader.load())
                
                if not docs:
                    st.error("Could not parse any text out of the uploaded files.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                    chunks = text_splitter.split_documents(docs)
                    
                    status_box.text(f"Generating vectors for {len(chunks)} chunks with gemini-embedding-2-preview...")
                    
                    # Core Active Production Embedding Pipeline
                    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview", google_api_key=google_api_key)
                    
                    # We switch directly to an In-Memory Vector Store to completely eliminate SQLite file lock errors
                    st.session_state["vector_store"] = InMemoryVectorStore.from_documents(chunks, embeddings)
                    
                    status_box.empty()
                    st.success(f"🎉 Successfully indexed {len(chunks)} chunks in RAM! Your assistant is armed and ready.")
                    
            except Exception as e:
                status_box.empty()
                st.error(f"❌ Processing error: {str(e)}")

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

# Helper to format pulled document context structures
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Main Application Logic
if not google_api_key:
    st.info("⚠️ Please enter your Google Gemini API Key in the sidebar or save it in Streamlit Secrets to begin.")
else:
    if st.session_state["vector_store"] is not None:
        retriever = st.session_state["vector_store"].as_retriever(search_kwargs={"k": 3})
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
        
        # Completely modern LCEL Chain pipeline replacing legacy 'create_retrieval_chain'
        rag_chain = (
            {"context": retriever | format_docs, "input": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        st.subheader("💬 Ask Your Assistant")
        user_query = st.text_input("Type your question or simulate voice command input:")
        
        if user_query:
            with st.spinner("Searching database and synthesizing audio reply..."):
                try:
                    answer = rag_chain.invoke(user_query)
                    
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
        st.warning("📥 Database empty. Please upload documents and click 'Build/Refresh Vector DB' in the sidebar to begin.")
