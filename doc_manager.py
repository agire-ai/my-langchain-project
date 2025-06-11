import os
import json
import hashlib

DOC_TRACK_FILE = "docs.json"

def file_to_hash(file_path):
    """Calculates the MD5 hash of a file's content."""
    with open(file_path, "rb") as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()

def load_docs_tracking():
    """Loads the document tracking data from a JSON file."""
    if not os.path.exists(DOC_TRACK_FILE):
        return {}
    try:
        with open(DOC_TRACK_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_docs_tracking(tracking):
    """Saves the document tracking data to a JSON file."""
    with open(DOC_TRACK_FILE, "w") as f:
        json.dump(tracking, f, indent=2)

def add_doc_tracking(doc_hash, doc_name):
    """
    Adds a document's info to the tracking file using its pre-calculated hash.
    
    Args:
        doc_hash: The MD5 hash of the document.
        doc_name: The original name of the document file.
    """
    tracking = load_docs_tracking()
    # The function now directly uses the provided hash
    tracking[doc_hash] = {
        "file_name": doc_name,
        "file_hash": doc_hash,
    }
    save_docs_tracking(tracking)
    return doc_hash

def remove_doc_tracking(doc_hash):
    """Removes a document's info from the tracking file."""
    tracking = load_docs_tracking()
    if doc_hash in tracking:
        del tracking[doc_hash]
        save_docs_tracking(tracking)
