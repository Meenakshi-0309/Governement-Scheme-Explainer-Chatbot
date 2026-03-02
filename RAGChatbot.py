import streamlit as st
import sqlite3
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# =====================================================
# DATABASE SETUP
# =====================================================
conn = sqlite3.connect("chat_logs.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_message TEXT,
    bot_response TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    user_name TEXT,
    scheme_name TEXT,
    opinion TEXT
)
""")
conn.commit()

# -------------------- DB FUNCTIONS --------------------
def save_conversation(user_msg, bot_msg):
    cursor.execute(
        "INSERT INTO conversations VALUES (NULL, ?, ?, ?)",
        (datetime.now().isoformat(), user_msg, bot_msg)
    )
    conn.commit()

def load_conversations():
    cursor.execute("SELECT user_message, bot_response FROM conversations ORDER BY id")
    return cursor.fetchall()

def clear_chat_history():
    cursor.execute("DELETE FROM conversations")
    conn.commit()

def save_feedback(name, scheme, opinion):
    cursor.execute(
        "INSERT INTO feedback VALUES (NULL, ?, ?, ?, ?)",
        (datetime.now().isoformat(), name, scheme, opinion)
    )
    conn.commit()

def load_feedbacks():
    cursor.execute(
        "SELECT id, timestamp, user_name, scheme_name, opinion FROM feedback ORDER BY id DESC"
    )
    return cursor.fetchall()

def delete_feedback(fid):
    cursor.execute("DELETE FROM feedback WHERE id = ?", (fid,))
    conn.commit()

# =====================================================
# LOAD VECTORSTORE
# =====================================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )

vectorstore = load_vectorstore()

# =====================================================
# LOAD LLM
# =====================================================
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    max_tokens=220
)

# =====================================================
# RAG RESPONSE FUNCTION (ENGLISH + TELUGU)
# =====================================================
def chatbot_response_rag(user_query):
    docs = vectorstore.similarity_search(user_query, k=4)

    if not docs:
        return "Information not available in the provided scheme documents."

    context = "\n\n".join(doc.page_content for doc in docs)

    system_prompt = """
You are a government scheme explainer chatbot.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT add external knowledge.
3. Do NOT copy text verbatim.
4. Summarize in your own words.
5. Keep answers short, clear, and factual.
6. Maximum 5 bullet points.
7. If the user asks in Telugu, respond in Telugu.
8. If the user asks in English, respond in English.
9. If information is missing, say:
   "Information not available in the document."
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"""
Context:
{context}

Question:
{user_query}

Answer:
""")
    ]

    return llm.invoke(messages).content.strip()

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(page_title="Government Scheme Explainer", layout="wide")

# ------------------ NAVIGATION ------------------
if "page" not in st.session_state:
    st.session_state.page = "Chatbot"

st.sidebar.title("Navigation")
st.session_state.page = st.sidebar.radio("Go to", ["Chatbot", "Feedback"])

# ------------------ STYLING ------------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#f4f7fb,#ffffff); font-family: 'Segoe UI'; }
.header { text-align:center; font-size:38px; font-weight:700; color:#1f2937; margin-top:15px; }
.disclaimer { 
    text-align:center; 
    font-size:14px; 
    color:#6b7280; 
    margin-bottom:25px; 
    max-width:900px; 
    margin-left:auto; 
    margin-right:auto;
}
.chat-container { max-width:900px; margin:auto; padding-bottom:160px; }
.chat-box { padding:16px; border-radius:14px; margin-bottom:12px; box-shadow:0 6px 16px rgba(0,0,0,.08); }
.user { background:#e0f2fe; border-left:6px solid #0284c7; }
.bot { background:#ffffff; border-left:6px solid #6b7280; }
.feedback-section { max-width:700px; margin:40px auto; background:#f9fafb; padding:20px; border-radius:14px; }
.feedback-item { background:white; padding:12px; border-radius:12px; margin-bottom:10px; }
</style>
""", unsafe_allow_html=True)

# =====================================================
# CHATBOT PAGE
# =====================================================
if st.session_state.page == "Chatbot":

    # ✅ HEADER
    st.markdown(
        "<div class='header'>🤖 Government Scheme Explainer Chatbot</div>",
        unsafe_allow_html=True
    )

    # ✅ DISCLAIMER (VISIBLE & CLEAR)
    st.markdown(
        """
        <div class="disclaimer">
        <em>
        This chatbot provides information based on publicly available government documents.
        <br>
        For official confirmation, please visit the official government website or contact authorities.
        </em>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='disclaimer'><em>Ask in English or Telugu (తెలుగు)</em></div>", unsafe_allow_html=True)

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_conversations()

    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    for u, b in st.session_state.chat_history:
        st.markdown(f"<div class='chat-box user'><b>You:</b> {u}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-box bot'><b>Bot:</b> {b}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([0.9, 0.1])
    with col1:
        user_message = st.chat_input("Ask about government schemes (English / తెలుగు)")
    with col2:
        if st.button("🗑️"):
            clear_chat_history()
            st.session_state.chat_history = []
            st.rerun()

    if user_message:
        reply = chatbot_response_rag(user_message)
        st.session_state.chat_history.append((user_message, reply))
        save_conversation(user_message, reply)
        st.rerun()

# =====================================================
# FEEDBACK PAGE
# =====================================================
if st.session_state.page == "Feedback":
    st.markdown("<div class='header'>💬 Scheme Feedback</div>", unsafe_allow_html=True)

    st.markdown("<div class='feedback-section'>", unsafe_allow_html=True)
    name = st.text_input("Your Name")
    scheme = st.text_input("Scheme Name")
    opinion = st.text_area("Your Experience / Opinion")

    if st.button("Submit Feedback"):
        if name and scheme and opinion:
            save_feedback(name, scheme, opinion)
            st.success("Feedback submitted")
            st.rerun()
        else:
            st.warning("Please fill all fields")

    st.markdown("---")
    st.subheader("Submitted Feedbacks")

    for fid, ts, n, s, o in load_feedbacks():
        st.markdown(f"""
        <div class='feedback-item'>
            <b>{s}</b> – {n}<br>
            {o}<br>
            <small>{ts}</small>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Delete", key=f"del_{fid}"):
            delete_feedback(fid)
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
