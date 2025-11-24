import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
import whisper
import numpy as np
from scipy.io import wavfile
import time
import tempfile

from dotenv import load_dotenv
load_dotenv()

# Load API Keys
os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

groq_api_key = os.getenv("GROQ_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Load Whisper model (cached)
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base")

# Create Vector Embeddings
def create_vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("🔄 Creating vector embeddings... This may take a moment."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            st.session_state.loader = PyPDFDirectoryLoader("papers")
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, 
                chunk_overlap=200
            )
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(
                st.session_state.docs[:50]
            )
            st.session_state.vectors = FAISS.from_documents(
                st.session_state.final_documents, 
                st.session_state.embeddings
            )

# Transcribe audio using Whisper
def transcribe_audio(audio_file):
    model = load_whisper_model()
    result = model.transcribe(audio_file)
    return result["text"]

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .css-1d391kg {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 10px;
    }
    h1 {
        color: #ffffff;
        text-align: center;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .subtitle {
        color: #ffffff;
        text-align: center;
        font-size: 1.2rem;
        margin-bottom: 2rem;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        border-radius: 25px;
        border: 2px solid #667eea;
        padding: 0.75rem 1.5rem;
    }
    .answer-box {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: #1a1a1a;
        font-size: 1.1rem;
        line-height: 1.6;
    }
    .info-box {
        background-color: rgba(102, 126, 234, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background-color: rgba(76, 175, 80, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1>🤖 RAG Document Q&A</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Powered by Groq, Qwen & Whisper AI</p>", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.markdown("### ⚙️ Model Settings")
    st.markdown("---")
    
    # Model Selection - Using Qwen
    engine = st.selectbox(
        "Select an AI model",
        ["qwen/qwen3-32b", "llama-3.1-8b-instant"],
        help="Choose the Groq model to use"
    )
    
    # Temperature Slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more creative, lower values more focused"
    )
    
    # Max Tokens Slider
    max_tokens = st.slider(
        "Max Tokens",
        min_value=50,
        max_value=2000,
        value=500,
        step=50,
        help="Maximum number of tokens to generate"
    )
    
    st.markdown("---")
    
    # Document Embedding Section
    st.markdown("### 📚 Document Processing")
    if st.button("🔨 Create Vector Database", use_container_width=True):
        create_vector_embedding()
        st.markdown("<div class='success-box'>✅ Vector Database is ready!</div>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Status")
    
    # Display current settings
    st.info(f"""
    **Current Settings:**
    - Model: `{engine}`
    - Temperature: `{temperature}`
    - Max Tokens: `{max_tokens}`
    """)
    
    if "vectors" in st.session_state:
        st.success("✅ Vector DB: Ready")
    else:
        st.warning("⚠️ Vector DB: Not initialized")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.info("""
    This app uses:
    - 🚀 Groq for fast inference
    - 🤖 Qwen & other models
    - 🎤 Whisper for voice input
    - 📄 FAISS for vector search
    - 🤗 HuggingFace embeddings
    """)

# Initialize LLM with selected parameters
llm = ChatGroq(groq_api_key=groq_api_key, model_name=engine, temperature=temperature, max_tokens=max_tokens)

# Main content area
st.markdown("<div class='info-box'>💡 Choose your input method: Type your question or use voice recording</div>", unsafe_allow_html=True)

# Create tabs for input methods
tab1, tab2 = st.tabs(["⌨️ Text Input", "🎤 Voice Input"])

user_prompt = None
text_prompt = None
voice_prompt = None

with tab1:
    st.markdown("### Type your question")
    text_prompt = st.text_input(
        "Enter your query from the research paper",
        placeholder="e.g., What is the main conclusion of the paper?",
        label_visibility="collapsed",
        key="text_input_field"
    )

with tab2:
    st.markdown("### Record your question")
    st.markdown("<div class='info-box'>🎙️ Click the record button below and speak your question</div>", unsafe_allow_html=True)
    
    audio_file = st.audio_input("Record your question")
    
    if audio_file is not None:
        with st.spinner("🔄 Transcribing audio..."):
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # Transcribe
            transcribed_text = transcribe_audio(tmp_file_path)
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            st.success("✅ Transcription complete!")
            st.markdown(f"**Transcribed text:** {transcribed_text}")
            voice_prompt = transcribed_text
            
            # Store in session state with timestamp
            st.session_state.last_voice_prompt = transcribed_text
            st.session_state.voice_prompt_time = time.time()

# Determine which prompt to use
if text_prompt:
    user_prompt = text_prompt
elif 'last_voice_prompt' in st.session_state and voice_prompt:
    user_prompt = voice_prompt

# Process query
if user_prompt:
    if "vectors" not in st.session_state:
        st.error("❌ Please create the vector database first using the sidebar button!")
    else:
        with st.spinner("🔍 Searching documents and generating answer..."):
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vectors.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            start = time.process_time()
            
            # OPTION 2: For streaming with real-time display
            response_container = st.empty()
            full_answer = ""
            
            # Stream the response
            for chunk in retrieval_chain.stream({'input': user_prompt}):
                if 'answer' in chunk:
                    full_answer += chunk['answer']
                    response_container.markdown(f"<div class='answer-box' style='color: #1a1a1a;'>{full_answer}</div>", unsafe_allow_html=True)
            
            response_time = time.process_time() - start
            
            # Store the final response for document display
            final_response = {'answer': full_answer, 'context': []}
            
            # For document context, you might need to run it again without streaming
            # or extract context from the retriever separately
            with st.expander("📄 View Source Documents"):
                st.markdown("**Relevant document chunks:**")
                # You might need to get documents separately
                docs = retriever.get_relevant_documents(user_prompt)
                for i, doc in enumerate(docs):
                    st.markdown(f"**Document Chunk {i+1}:**")
                    st.info(doc.page_content)
                    st.markdown("---")
            
            st.markdown(f"<small>⏱️ Response time: {response_time:.2f} seconds</small>", unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #ffffff;'>Made with ❤️ using Streamlit, LangChain & Whisper</p>", 
    unsafe_allow_html=True
)