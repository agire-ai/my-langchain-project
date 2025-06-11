from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List

def load_file_as_documents(file_path):
    """Dispatches to the appropriate loader based on file type."""
    ext = file_path.lower().split(".")[-1]
    if ext == "pdf":
        loader = PyPDFLoader(file_path)
    elif ext == "txt":
        loader = TextLoader(file_path, encoding="utf-8")
    elif ext == "docx":
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return loader.load()

def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    file_hash: str = None,
    file_name: str = None
) -> List[Document]:
    """
    Split Documents into smaller chunks and add rich metadata.
    
    Args:
        documents: List of documents to be split.
        chunk_size: The size of each chunk.
        chunk_overlap: The overlap between chunks.
        file_hash: The hash of the source file to add to metadata.
        file_name: The name of the source file to add to metadata.
        
    Returns:
        A list of split documents with enhanced metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    split_docs = splitter.split_documents(documents)

    # Add rich metadata to each chunk
    for i, doc in enumerate(split_docs):
        doc.metadata["chunk_id"] = i
        doc.metadata["source"] = doc.metadata.get("source", "unknown")
        doc.metadata["page"] = doc.metadata.get("page", "unknown")
        if file_hash:
            doc.metadata["file_hash"] = file_hash
        if file_name:
            doc.metadata["file_name"] = file_name

    return split_docs

if __name__ == "__main__":
    pdf_path = "form10-k.pdf"  # Replace with your test PDF path

    print(f"Loading PDF: {pdf_path}")
    documents = load_file_as_documents(pdf_path)

    print("Splitting documents...")
    # Example of how to call the updated function
    split_docs = split_documents(
        documents,
        file_hash="dummy_hash_12345",
        file_name="test_document.pdf"
    )
    print(f"Generated {len(split_docs)} chunks.")

    print("Metadata for the first chunk:")
    print(split_docs[0].metadata)
