# LocalVoiceIntel 🎙️ (Gemini Free Edition)

An intelligent mini-voice assistant that processes requests, searches local document knowledge structures using Retrieval-Augmented Generation (RAG), and responds back immediately using synthesized speech audio files. Powered entirely through the cost-free Google Gemini API tier.

---

## 🛠️ Tech Stack
* **Frontend UI:** Streamlit
* **Orchestration Framework:** LangChain
* **Vector DB Engine:** ChromaDB
* **LLM Core & Embeddings:** Google AI Studio (`gemini-2.5-flash` & `text-embedding-004`)
* **Text-to-Speech:** gTTS (Google Text-to-Speech)

---

## 🔒 Automated Key Secrets Setup (Streamlit Cloud)
To completely avoid typing your API key every time you access the live platform:
1. Access your active **Streamlit Dashboard**.
2. Click the app menu button (**...**) -> **Settings** -> **Secrets**.
3. Create a parameter entry mapping like this:
   ```toml
   GOOGLE_API_KEY = "AIzaSy..."
