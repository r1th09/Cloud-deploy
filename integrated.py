"""
integrated_app.py

Single-file Streamlit app that contains:
- The Streamlit frontend (UI)
- The FastAPI backend logic (AIAssistant), adapted to run inside Streamlit
- Pydantic models inlined from models.py

Run:
    streamlit run integrated_app.py

Requirements:
    streamlit, google-generativeai, python-dotenv, pydantic

Important:
- Set GEMINI_API_KEY in environment before running (e.g. in .env or Streamlit Cloud Secrets).
"""

import os
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
import re
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
    due_date: Optional[datetime] = None
    priority: Priority = Priority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assignee: Optional[str] = None
    source_platform: Platform
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
    action_items: List[str] = []
    tasks: List[Task] = []
    priority: Priority = Priority.NORMAL
    suggested_replies: List[str] = []
    reminders: List[Dict[str, Any]] = []
    context_updates: Optional[Dict[str, Any]] = None


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
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}

    def _categorize_message(self, content: str, platform: Platform) -> str:
        content_lower = content.lower()
        if any(w in content_lower for w in ["urgent", "asap", "emergency"]):
            return "urgent"
        elif any(w in content_lower for w in ["follow up", "follow-up", "followup"]):
            return "follow-up"
        elif any(w in content_lower for w in ["task", "todo", "to-do"]):
            return "task"
        return "normal"

    def _extract_tasks(self, content: str, platform: Platform) -> List[Task]:
        tasks = []
        if "task:" in content.lower() or "todo:" in content.lower():
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

The user has received this message:
"{content}"

Generate **exactly three concise, polite, and professional replies** suitable for responding to this message.
Each reply should be one short paragraph (under 30 words).

Return them in the following numbered format:
1. <first reply>
2. <second reply>
3. <third reply>
"""
            response = self.model.generate_content(prompt)
            text = getattr(response, "text", None) or str(response)

            # Extract numbered replies (1., 2., 3.)
            replies = re.findall(r"\d+\.\s*(.+)", text)
            cleaned = [r.strip().strip('"').strip("'") for r in replies if len(r.strip()) > 2]

            if len(cleaned) < 3:
                cleaned = [s.strip() for s in text.split("\n") if s.strip()][:3]

            return cleaned[:3]
        except Exception:
            return [
                "Thanks for your message. I'll get back to you shortly.",
                "Understood. I’ll review and respond soon.",
                "Received your note. Expect a follow-up soon."
            ]

    async def process_message(self, message: Message) -> ResponseModel:
        self.conversation_history.setdefault(message.sender, []).append({
            "content": message.content,
            "timestamp": message.timestamp
        })

        message.category = self._categorize_message(message.content, message.platform)
        tasks = self._extract_tasks(message.content, message.platform)

        platform_prompts = {
            Platform.EMAIL: "You are an email assistant. Help prioritize and suggest responses for emails.",
            Platform.SLACK: "You are a Slack assistant. Help summarize and identify action items.",
            Platform.WHATSAPP: "You are a WhatsApp assistant. Help manage communications efficiently."
        }
        system_prompt = platform_prompts.get(message.platform, "")

        context = f"""
{system_prompt}

From: {message.sender}
Category: {message.category}
Priority: {message.priority}
Message: {message.content}

Please provide:
1. A brief summary of the message
2. A suggested response
3. Action items or follow-ups
4. Any reminders needed
"""

        ai_response = self.model.generate_content(context)
        ai_text = getattr(ai_response, "text", None) or str(ai_response)
        suggested_replies = self._generate_suggested_replies(message.content)

        reminders = []
        if message.category in ["urgent", "follow-up"]:
            reminders.append({
                "type": message.category,
                "due_date": datetime.now().isoformat(),
                "message": f"Follow up with {message.sender}: {message.content[:100]}..."
            })

        return ResponseModel(
            content=ai_text,
            summary=f"Message from {message.sender} - {message.category.upper()}",
            action_items=[f"Reply to {message.sender}"] if message.category == "urgent" else [],
            tasks=tasks,
            priority=message.priority,
            suggested_replies=suggested_replies,
            reminders=reminders
        )


assistant = AIAssistant()

def process_message_sync(message: Message) -> ResponseModel:
    return asyncio.run(assistant.process_message(message))


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Communication Assistant", page_icon="💬", layout="wide")

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
    sender = st.text_input("Sender", placeholder="Enter sender's name or email")
    st.markdown("---")
    st.metric("Messages Processed", len(st.session_state.chat_history))
    st.metric("Active Tasks", sum(len(c['response'].get('tasks', [])) for c in st.session_state.chat_history))

col1, col2 = st.columns([3, 4])

with col1:
    st.subheader("New Message")
    message = st.text_area("Type your message", placeholder="Type your message here...", height=150)

    if st.button("📤 Send Message", type="primary", disabled=not (message and sender)):
        with st.spinner("Processing message..."):
            try:
                msg = Message(content=message, platform=platform, sender=sender, priority=priority)
                resp = process_message_sync(msg)

                payload = {
                    "content": resp.content,
                    "summary": resp.summary,
                    "action_items": resp.action_items,
                    "tasks": [t.model_dump() for t in resp.tasks],
                    "priority": resp.priority.value,
                    "suggested_replies": resp.suggested_replies,
                    "reminders": resp.reminders
                }

                st.session_state.current_message = {
                    "message": msg.model_dump(),
                    "response": payload,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.chat_history.append(st.session_state.current_message)
                st.success("✅ Message processed successfully!")
            except Exception as e:
                st.error(f"Error: {e}")

with col2:
    st.subheader("Response & Analysis")
    if st.session_state.current_message:
        response = st.session_state.current_message['response']
        st.markdown(f"**Priority:** {response.get('priority', '').upper()}")
        if response.get('summary'):
            st.markdown(f"**Summary:** {response['summary']}")
        if response.get('tasks'):
            st.markdown("#### 📋 Tasks")
            for task in response['tasks']:
                with st.expander(task['title'], expanded=False):
                    st.write(f"**Description:** {task['description']}")
                    st.write(f"**Priority:** {task['priority']}")
        if response.get('suggested_replies'):
            st.markdown("#### 💬 Suggested Replies")
            for reply in response['suggested_replies']:
                st.code(reply)
        if response.get('reminders'):
            st.markdown("#### ⏰ Reminders")
            for rem in response['reminders']:
                st.info(f"🔔 {rem['message']} — Due: {rem['due_date']}")

# History
st.markdown("---")
st.subheader("📝 Message History")
for chat in reversed(st.session_state.chat_history):
    msg = chat['message']
    with st.expander(f"{msg['platform'].upper()} | {msg['sender']} | {chat['timestamp']}"):
        st.markdown(f"**Message:** {msg['content']}")
        resp = chat['response']
        if resp.get('suggested_replies'):
            st.markdown("**Suggested Replies:**")
            for reply in resp['suggested_replies']:
                st.markdown(f"- {reply}")

st.markdown("<hr><center>AI Communication Assistant — Powered by Gemini</center>", unsafe_allow_html=True)
