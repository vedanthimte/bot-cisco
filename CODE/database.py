# database.py - Handles persistent data for all users

import sqlite3
from typing import List, Dict

DB_PATH = 'db/prmitr_cisco.db'

def init_db():
    """Create the necessary tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table for user roles (You can add more students here later)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            name TEXT,
            email TEXT,
            hashed_password TEXT,
            role TEXT
        )
    """)
    
    # Table for permanent chat history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    conn.close()

def save_message(username: str, role: str, content: str):
    """Saves a single message to the permanent history."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO chat_history (username, role, content) VALUES (?, ?, ?)",
        (username, role, content)
    )
    conn.commit()
    conn.close()

def load_messages(username: str) -> List[Dict]:
    """Loads all chat messages for a specific user."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM chat_history WHERE username = ? ORDER BY timestamp",
        (username,)
    )
    # Convert tuples to dictionary format for Streamlit
    messages = [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]
    conn.close()
    return messages