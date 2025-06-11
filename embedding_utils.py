# embedding_utils.py

import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from typing import List

def embed_documents(documents: List[Document], embedding_model: OpenAIEmbeddings) -> Chroma:
    """Embed Documents and return a Chroma vectorstore."""
    vectorstore = Chroma.from_documents(documents, embedding_model)
    return vectorstore

def create_or_load_chroma(
    documents: List[Document],
    embedding_model: OpenAIEmbeddings,
    persist_directory: str = "./chroma_db"
) -> Chroma:
    """
    Load existing Chroma vectorstore from disk, or create it from documents if not present.

    Args:
        documents: List of Documents to embed if needed.
        embedding_model: The embedding model to use.
        persist_directory: Directory where Chroma vectorstore is stored.

    Returns:
        Chroma vectorstore instance.
    """

    # If vectorstore exists, load it
    if os.path.exists(persist_directory) and os.path.isdir(persist_directory):
        print(f"Loading existing Chroma vectorstore from {persist_directory}...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
    else:
        print(f"Creating new Chroma vectorstore at {persist_directory}...")
        vectorstore = Chroma.from_documents(
            documents,
            embedding_model,
            persist_directory=persist_directory
        )
    return vectorstore


# Optional test when run standalone
if __name__ == "__main__":
    from pdf_utils import load_pdf_as_documents, split_documents
    import os
    from dotenv import load_dotenv

    load_dotenv()

    pdf_path = "form10-k.pdf"  # Replace with your test PDF path

    print(f"Loading PDF: {pdf_path}")
    documents = load_pdf_as_documents(pdf_path)
    print(f"Loaded {len(documents)} documents.")

    print("Splitting documents...")
    split_docs = split_documents(documents)
    print(f"Generated {len(split_docs)} chunks.")

    print("Embedding chunks...")
    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = embed_documents(split_docs, embedding_model)
    print(f"✅ Embedded {len(split_docs)} chunks into vectorstore.")

    # Optional: test retrieval
    query = "What is the company's revenue?"
    retriever = vectorstore.as_retriever()
    retrieved_docs = retriever.invoke(query)

    print(f"Retrieved {len(retrieved_docs)} document(s) for query: {query}")
    for i, doc in enumerate(retrieved_docs):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:1000])
