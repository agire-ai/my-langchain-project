import streamlit as st
from dotenv import load_dotenv
import os
import shutil
import hashlib
import time
import gc  # Import the garbage collection module
from pdf_utils import load_file_as_documents, split_documents
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from doc_manager import load_docs_tracking, add_doc_tracking, remove_doc_tracking

# --- Page Configuration ---
st.set_page_config(page_title="Analyseur de Documents Financiers", layout="wide")

# --- Environment and API Key Setup ---
load_dotenv(override=True)

# --- Constants ---
PERSIST_DIRECTORY = "./chroma_db_main"
DOC_TRACK_FILE = "docs.json"

# --- Session State Initialization ---
if "history" not in st.session_state:
    st.session_state.history = []

# --- Helper Functions ---
def file_to_hash(file_bytes):
    """Generates an MD5 hash from the file's bytes."""
    return hashlib.md5(file_bytes).hexdigest()

# --- Initialize Vectorstore ---
@st.cache_resource
def get_vectorstore(api_key):
    """Initializes and returns the Chroma vectorstore."""
    if not api_key:
        return None
    embedding_model = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_model
    )
    return vectorstore

# --- Sidebar ---
st.sidebar.title("🛠️ Configuration")

# 1. API Key Input
api_key_input = st.sidebar.text_input(
    'Clé API OpenAI:',
    type='password',
    help="Votre clé API OpenAI ne sera pas stockée.",
    key="api_key_input"
)

# Use provided key or key from .env file
if api_key_input:
    st.session_state.api_key = api_key_input
elif os.getenv("OPENAI_API_KEY"):
    st.session_state.api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is available
if 'api_key' not in st.session_state or not st.session_state.api_key:
    st.info("Veuillez entrer votre clé API OpenAI dans la barre latérale pour commencer.")
    st.stop()

# Initialize vectorstore only if API key is present
vectorstore = get_vectorstore(st.session_state.api_key)

# 2. Advanced Parameters
with st.sidebar.expander("Paramètres avancés"):
    chunk_size = st.number_input('Taille des morceaux (chunk size):', min_value=100, max_value=4096, value=512)
    k_results = st.number_input("Nombre de résultats (k):", min_value=1, max_value=20, value=3)

st.sidebar.divider()

st.sidebar.title("🗂️ Corpus de Documents")
tracking_data = load_docs_tracking()

if not tracking_data:
    st.sidebar.info("Aucun document n'a encore été indexé.")
else:
    st.sidebar.write("Documents Indexés:")
    for doc_hash, doc_info in tracking_data.items():
        with st.sidebar.expander(f"📄 {doc_info['file_name']}"):
            st.markdown(f"**Hash:** `{doc_hash[:8]}`")
            if st.button("🗑️ Supprimer", key=f"remove_{doc_hash}", use_container_width=True):
                ids_to_delete = vectorstore.get(where={"file_hash": doc_hash}).get('ids', [])
                if ids_to_delete:
                    vectorstore.delete(ids=ids_to_delete)
                    vectorstore.persist()
                remove_doc_tracking(doc_hash)
                st.success(f"'{doc_info['file_name']}' a été retiré du corpus.")
                st.rerun()

# --- Main App Interface ---
st.title("📚 Analyseur de Documents Financiers")
st.info("Téléversez de nouveaux documents pour les ajouter au corpus. La Q&A recherchera dans tous les documents indexés.")

uploaded_file = st.file_uploader("Téléverser un nouveau document", type=["pdf", "txt", "docx"])

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    doc_hash = file_to_hash(file_bytes)

    if st.session_state.get('processed_hash') == doc_hash:
        st.success(f"✅ '{uploaded_file.name}' a été indexé avec succès.")
    elif doc_hash in tracking_data:
        st.warning(f"⚠️ Ce document, '{uploaded_file.name}', était déjà dans le corpus.")
    else:
        st.info(f"Traitement de '{uploaded_file.name}'...")
        temp_file_path = f"temp_{uploaded_file.name}"
        with open(temp_file_path, "wb") as f:
            f.write(file_bytes)
        
        with st.spinner("Lecture, découpage et création des embeddings..."):
            documents = load_file_as_documents(temp_file_path)
            split_docs = split_documents(documents, chunk_size=chunk_size, chunk_overlap=50, file_hash=doc_hash, file_name=uploaded_file.name)
            
            st.info(f"📄 {len(split_docs)} morceaux (chunks) ont été créés.")
            
            if split_docs:
                vectorstore.add_documents(split_docs)
                vectorstore.persist()
                add_doc_tracking(doc_hash, uploaded_file.name)
                os.remove(temp_file_path)
                
                st.session_state.processed_hash = doc_hash
                st.rerun()
            else:
                st.error("Le document n'a pas pu être traité.")
                os.remove(temp_file_path)
else:
    if 'processed_hash' in st.session_state:
        del st.session_state['processed_hash']

# --- Q&A and Chat History Section ---
st.header("💬 Chat et Q&A")

# Display chat history
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input for new question
if query := st.chat_input("Posez une question sur l'ensemble des documents..."):
    # Add user query to history
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            # Create the QA chain
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, api_key=st.session_state.api_key)
            retriever = vectorstore.as_retriever(search_kwargs={"k": k_results})
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=retriever,
                return_source_documents=True
            )
            
            # Get the answer from the LLM
            result = qa_chain.invoke(query)
            answer = result['result']
            st.markdown(answer)

            # Display source documents in an expander
            with st.expander("Documents sources utilisés"):
                for i, doc in enumerate(result['source_documents']):
                    metadata = doc.metadata
                    file_name = metadata.get("file_name", "Inconnu")
                    page = metadata.get("page", "N/A")
                    st.markdown(f"**Source:** `{file_name}` (Page: {page})")
                    # FIX: Add the loop index 'i' to the key to ensure uniqueness
                    st.text_area("", doc.page_content, height=150, key=f"source_{i}_{hash(doc.page_content)}")

    # Add assistant response to history
    st.session_state.history.append({"role": "assistant", "content": answer})
