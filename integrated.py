"""
integrated_app.py — Final Combined Version

✅ Restored original frontend (UI)
✅ Added "tonality" detection in backend
✅ Keeps 3 concise suggested replies
✅ Works locally and on Streamlit Cloud

Run:
    streamlit run integrated_app.py
"""

import os
import re
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv(override=True)

# ---------------------------
# Models
# ---------------------------
class Platform(str, Enum):
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    SLACK = "slack"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    FOLLOW_UP = "follow_up"

class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class Task(BaseModel):
    title: str
    description: str
    source_platform: Platform
    due_date: Optional[datetime] = None
    priority: Priority = Priority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assignee: Optional[str] = None
    created_at: datetime = datetime.now()

class Message(BaseModel):
    platform: Platform
    content: str
    sender: str
    priority: Priority = Priority.NORMAL
    timestamp: datetime = datetime.now()
    category: Optional[str] = None
    requires_action: bool = False
    context: Optional[Dict[str, Any]] = None

class ResponseModel(BaseModel):
    content: str
    summary: Optional[str] = None
    tone: Optional[str] = None
    action_items: List[str] = []
    tasks: List[Task] = []
    priority: Priority = Priority.NORMAL
    suggested_replies: List[str] = []
    reminders: List[Dict[str, Any]] = []


# ---------------------------
# AIAssistant
# ---------------------------
class AIAssistant:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set. Please configure it before running.")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.history: Dict[str, List[Dict[str, Any]]] = {}

    def _categorize_message(self, content: str) -> str:
        c = content.lower()
        if any(w in c for w in ["urgent", "asap", "emergency"]):
            return "urgent"
        elif any(w in c for w in ["follow up", "follow-up", "followup"]):
            return "follow-up"
        elif any(w in c for w in ["task:", "todo:", "to-do"]):
            return "task"
        return "normal"

    def _extract_tasks(self, content: str, platform: Platform) -> List[Task]:
        tasks = []
        for line in content.split("\n"):
            if "task:" in line.lower() or "todo:" in line.lower():
                title = line.split(":", 1)[1].strip()
                tasks.append(Task(
                    title=title,
                    description=title,
                    source_platform=platform
                ))
        return tasks

    def _generate_suggested_replies(self, content: str) -> List[str]:
        try:
            prompt = f"""
You are a professional communication assistant.
The user received this message:
"{content}"

Generate **exactly three concise, polite, and professional replies** under 25 words each.
Format:
1. ...
2. ...
3. ...
"""
            response = self.model.generate_content(prompt)
            text = getattr(response, "text", None) or str(response)
            replies = re.findall(r"\d+\.\s*(.+)", text)
            cleaned = [r.strip().strip('"') for r in replies if len(r.strip()) > 2]
            return cleaned[:3] or [
                "Thanks for your message. I'll get back to you shortly.",
                "Understood. I’ll review and respond soon.",
                "Received your note. Expect a follow-up soon."
            ]
        except Exception:
            return [
                "Thanks for your message. I'll get back to you shortly.",
                "Understood. I’ll review and respond soon.",
                "Received your note. Expect a follow-up soon."
            ]

    async def process_message(self, message: Message) -> ResponseModel:
        self.history.setdefault(message.sender, []).append({
            "content": message.content,
            "timestamp": message.timestamp
        })

        message.category = self._categorize_message(message.content)
        tasks = self._extract_tasks(message.content, message.platform)

        platform_prompt = {
            Platform.EMAIL: "You are an email communication assistant.",
            Platform.SLACK: "You are a Slack assistant helping summarize professional discussions.",
            Platform.WHATSAPP: "You are a WhatsApp assistant for daily communication management."
        }.get(message.platform, "")

        context = f"""
{platform_prompt}

From: {message.sender}
Category: {message.category}
Priority: {message.priority}
Message: {message.content}

Please analyze this message and provide:
1. A short summary of the message
2. The tone or emotion of the message (e.g., polite, urgent, angry, formal)
3. Key action items or follow-ups (if any)
4. One appropriate response text
"""

        ai_response = self.model.generate_content(context)
        ai_text = getattr(ai_response, "text", None) or str(ai_response)

        tone_match = re.search(r"(Tone|Emotion):\s*(.+)", ai_text, re.IGNORECASE)
        tone = tone_match.group(2).strip() if tone_match else "Not specified"

        suggested_replies = self._generate_suggested_replies(message.content)
        reminders = []
        if message.category in ["urgent", "follow-up"]:
            reminders.append({
                "type": message.category,
                "due_date": datetime.now().isoformat(),
                "message": f"Follow up with {message.sender}: {message.content[:80]}..."
            })

        return ResponseModel(
            content=ai_text.strip(),
            summary=f"Message from {message.sender} ({message.category.upper()})",
            tone=tone,
            tasks=tasks,
            priority=message.priority,
            suggested_replies=suggested_replies,
            reminders=reminders
        )


assistant = AIAssistant()

def process_message_sync(message: Message) -> ResponseModel:
    return asyncio.run(assistant.process_message(message))


# ---------------------------
# Streamlit UI (Old Design Restored)
# ---------------------------
st.set_page_config(page_title="AI Communication Assistant", page_icon="💬", layout="wide")

# Custom CSS
st.markdown("""
<style>
.stButton>button { width: 100%; margin-top: 1rem; }
.message-box { padding: 1rem; border-radius: 0.5rem; background-color: #f0f2f6; margin-bottom: 1rem; }
.urgent { border-left: 4px solid #ff4b4b; }
.high { border-left: 4px solid #ffa500; }
.normal { border-left: 4px solid #00bfff; }
.low { border-left: 4px solid #32cd32; }
.follow_up { border-left: 4px solid #9370db; }
</style>
""", unsafe_allow_html=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_message' not in st.session_state:
    st.session_state.current_message = None

st.title("💬 AI Communication Assistant")

# Sidebar
with st.sidebar:
    st.header("Message Settings")
    platform = st.selectbox("Platform", [Platform.EMAIL, Platform.SLACK, Platform.WHATSAPP])
    priority = st.selectbox("Priority", [Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW, Priority.FOLLOW_UP])
    sender = st.text_input("Sender", placeholder="Enter sender's email or name")
    st.markdown("---")
    st.metric("Messages Processed", len(st.session_state.chat_history))
    st.metric("Active Tasks", sum(len(c['response'].get('tasks', [])) for c in st.session_state.chat_history))

# Columns
col1, col2 = st.columns([3, 4])

with col1:
    st.subheader("New Message")
    msg_text = st.text_area("Type your message", placeholder="Type your message here...", height=150)
    if st.button("📤 Send Message", type="primary", disabled=not (msg_text and sender)):
        with st.spinner("Processing..."):
            try:
                msg = Message(content=msg_text, platform=platform, sender=sender, priority=priority)
                resp = process_message_sync(msg)
                payload = {
                    "content": resp.content,
                    "summary": resp.summary,
                    "tone": resp.tone,
                    "tasks": [t.model_dump() for t in resp.tasks],
                    "priority": resp.priority.value,
                    "suggested_replies": resp.suggested_replies,
                    "reminders": resp.reminders
                }
                st.session_state.current_message = {"message": msg.model_dump(), "response": payload, "timestamp": datetime.now().isoformat()}
                st.session_state.chat_history.append(st.session_state.current_message)
                st.success("✅ Message processed successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.subheader("Response & Analysis")
    if st.session_state.current_message:
        r = st.session_state.current_message['response']
        st.markdown(f"**Priority:** {r.get('priority', '').upper()}")
        if r.get("tone"):
            st.markdown(f"**Tone:** {r['tone']}")
        if r.get('summary'):
            st.markdown(f"**Summary:** {r['summary']}")
        if r.get('suggested_replies'):
            st.markdown("#### 💬 Suggested Replies")
            for rep in r['suggested_replies']:
                st.code(rep)
        if r.get('reminders'):
            st.markdown("#### ⏰ Reminders")
            for rem in r['reminders']:
                st.info(f"🔔 {rem['message']} — Due: {rem['due_date']}")

# History
st.markdown("---")
st.subheader("📝 Message History")
for chat in reversed(st.session_state.chat_history):
    m = chat['message']
    with st.expander(f"{m['platform'].upper()} | {m['sender']} | {chat['timestamp']}"):
        st.markdown(f"**Message:** {m['content']}")
        resp = chat['response']
        if resp.get('tone'):
            st.markdown(f"**Tone:** {resp['tone']}")
        if resp.get('suggested_replies'):
            st.markdown("**Suggested Replies:**")
            for rep in resp['suggested_replies']:
                st.markdown(f"- {rep}")

st.markdown("<hr><center>AI Communication Assistant — Powered by Gemini</center>", unsafe_allow_html=True)
