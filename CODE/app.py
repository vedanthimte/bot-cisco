# app.py

import streamlit as st
import os
from dotenv import load_dotenv

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings 
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding 
from llama_index.core.tools import FunctionTool
from llama_index.core.tools import QueryEngineTool 
# REMOVED: from llama_index.core.agent import ReActAgent  <-- Removed problematic import
from llama_index.core.chat_engine import CondenseQuestionChatEngine # Universal Stable Engine

# Custom Imports
from users import USERS 
import database as db  
import tools as custom_tools 

# --- 1. CONFIGURATION & ENVIRONMENT SETUP ---

load_dotenv() 

try:
    API_KEY = os.getenv("GEMINI_API_KEY") 
    
    if not API_KEY:
         raise ValueError("GEMINI_API_KEY not found in environment or .env file.")
    
    os.environ["GEMINI_API_KEY"] = "AIzaSyByJbj54PsMhsr2npl2XjsB3Z8YsKsafns"
    db.init_db()
    
except Exception as e:
    st.error(f"🚨 Initialization Error. Please check your environment variables and file paths. Details: {e}")
    st.stop()


# --- 2. AUTHENTICATION & AUTHORIZATION (Simple Login) ---

def login_form():
    """Renders the simple login form for Admin/Student access."""
    st.set_page_config(page_title="PRMITR Cisco Academy Login")
    st.title("PRMITR Cisco Academy Login")
    
    with st.form("login_form"):
        username_input = st.text_input("Username").lower()
        password_input = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

        if submitted:
            if username_input in USERS and USERS[username_input]['password'] == password_input:
                user_info = USERS[username_input]
                st.session_state["logged_in"] = True
                st.session_state["username"] = username_input
                st.session_state["name"] = user_info["name"]
                st.session_state["user_role"] = user_info["role"]
                st.rerun() 
            else:
                st.error("Invalid Username or Password.")

if "logged_in" not in st.session_state or not st.session_state["logged_in"]:
    login_form()
    st.stop()


# --- 3. UNIVERSAL CHAT ENGINE SETUP (STABLE) ---

def setup_engine(user_role: str):
    """Initializes the Universal ChatEngine, conditionally adding Admin tools."""
    
    model_name = "gemini-2.5-flash" 
    
    # 1. Define LLM and EMBEDDING Model 
    llm_instance = Gemini(
        model=model_name, 
        temperature=0.1,
        system_instruction="You are a helpful academic assistant for the PRMITR Cisco Networking Academy. Your responses are concise and accurate. For factual questions, use your available tools."
    ) 
    embed_model_instance = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # 2. Set the global LlamaIndex Settings (Used by RAG and tools.py)
    Settings.llm = llm_instance
    Settings.embed_model = embed_model_instance

    # 3. Load Base RAG Documents 
    try:
        reader = SimpleDirectoryReader(input_dir="./data")
        documents = reader.load_data()
        base_index = VectorStoreIndex.from_documents(documents) 
    except Exception:
        base_index = VectorStoreIndex.from_documents([]) 
    
    base_query_engine = base_index.as_query_engine(streaming=True)
    
    # 4. Define Base RAG Tool 
    base_tool = QueryEngineTool.from_defaults(
        query_engine=base_query_engine,
        name="CCNA_Course_Information",
        description="Use this tool to answer all questions about the CCNA course schedule, fees, and module details from official documentation."
    )
    tools = [base_tool]
    
    # --- ADMIN TOOL INTEGRATION (CONDITIONAL) ---
    if user_role == "admin":
        # Conditionally define and add Admin tools
        learn_tool = FunctionTool.from_defaults(fn=custom_tools.record_admin_note)
        notes_query_engine = custom_tools.NOTES_INDEX.as_query_engine(streaming=True)
        notes_tool = QueryEngineTool.from_defaults(
            query_engine=notes_query_engine,
            name="Admin_Notes_Query",
            description="Use this tool to retrieve any information the Admin has previously asked the system to remember (e.g., new policies, future plans)."
        )
        tools.extend([learn_tool, notes_tool])
        
    # 5. Return the stable CondenseQuestionChatEngine for ALL users, 
    # but pass the full toolset to the Admin's engine.
    return CondenseQuestionChatEngine.from_defaults(
        query_engine=base_query_engine, 
        llm=llm_instance,
        tools=tools, # Pass tools (Admin gets 3, Student gets 1)
        streaming=True  
    )


# --- 4. MAIN APP EXECUTION ---

# Access user information
username = st.session_state["username"]
user_role = st.session_state["user_role"]
name = st.session_state["name"]

# Setup the appropriate engine/agent
agent_or_engine = setup_engine(user_role) 
model_name = Settings.llm.model 

st.set_page_config(page_title="PRMITR Cisco Networking Assistant", layout="wide")
st.title("🤖 PRMITR Cisco Networking Academy Assistant")

# Sidebar UI elements
with st.sidebar:
    st.subheader(f"Welcome, {name}!")
    st.markdown(f"**Role:** `{user_role}` | **LLM:** `{model_name}`")
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()
    
    if user_role == "admin":
        st.divider()
        st.header("Admin Tools ⚙️")
        st.info("Dynamic Learning is active. Example: 'Please learn that the new lab hours are 9-5 starting next week.'")


# --- 5. CHAT HISTORY INITIALIZATION (Permanent Memory) ---

if "messages" not in st.session_state:
    st.session_state.messages = db.load_messages(username)
    
    if not st.session_state.messages:
        initial_greeting = (
            f"Hello {name}! I'm the **PRMITR Cisco Networking Assistant** (powered by {model_name}). "
            f"As a **{user_role}**, I can answer course questions quickly. "
            f"Admin: try asking me to **learn** a new policy."
        )
        st.session_state.messages.append({"role": "assistant", "content": initial_greeting})
        db.save_message(username, "assistant", initial_greeting)


# --- 6. CHAT INTERFACE AND MAIN LOOP ---

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask about the CCNA course, schedule, or ask me to learn something..."):
    # 1. Display User Message & Save to DB
    st.session_state.messages.append({"role": "user", "content": prompt})
    db.save_message(username, "user", prompt)
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get and Display Agent Response
    with st.chat_message("assistant"):
        with st.spinner(f"The Agent is thinking with {model_name}..."):
            
            # FINAL FIX: Use the universal stream_chat() method
            try:
                response = agent_or_engine.stream_chat(prompt) 
                
                # Use the .response_gen attribute for the iterable stream
                full_response = st.write_stream(response.response_gen) 

            except Exception as e:
                # Catch general exceptions (e.g., API issues, Index errors)
                error_msg = f"❌ **Critical Agent Error:** The response process crashed. Details: {e}"
                st.error(error_msg)
                full_response = error_msg


    # 3. Save Agent Response to DB and Session State
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    db.save_message(username, "assistant", full_response)