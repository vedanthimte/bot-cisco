# tools.py
from llama_index.core import VectorStoreIndex, Document
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
import os

# --- 1. CONFIGURATION FOR TOOLS (Executed on Import) ---

# This ensures the correct embedding model is configured before the index is loaded
try:
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
except Exception as e:
    print(f"CRITICAL ERROR in tools.py: Could not initialize embedding model: {e}")


# --- 2. ADMIN NOTES INDEX SETUP ---
NOTES_INDEX_DIR = "./admin_notes_index"

def init_notes_index():
    """Initializes a separate index for admin-provided notes."""
    try:
        if os.path.exists(NOTES_INDEX_DIR):
            return VectorStoreIndex.load_from_storage(
                index_path=NOTES_INDEX_DIR
            )
        else:
            return VectorStoreIndex.from_documents([], show_progress=True)
    except Exception:
        # If loading fails (e.g., corrupt index), create a new empty one
        return VectorStoreIndex.from_documents([], show_progress=True)

# The persistent index object, initialized immediately on import
NOTES_INDEX = init_notes_index()

# --- 3. FUNCTION TOOL DEFINITION ---

def record_admin_note(topic: str, content: str) -> str:
    """
    Function Tool: Records new, important information provided by the Admin 
    into the persistent knowledge base (NOTES_INDEX).
    
    Args:
        topic (str): The subject or topic of the note (e.g., "CCNA Fees 2026").
        content (str): The detailed, full text information to be stored.

    Returns:
        str: A confirmation message to be presented to the user.
    """
    global NOTES_INDEX

    new_document = Document(
        text=content, 
        metadata={"topic": topic, "source": "Admin Provided Note"}
    )
    
    NOTES_INDEX.insert(new_document)
    
    # Save the index to disk to ensure persistence across sessions
    NOTES_INDEX.storage_context.persist(persist_dir=NOTES_INDEX_DIR)

    return f"Confirmation: I have permanently recorded new information on the topic '{topic}' for future reference. You can now query this note."