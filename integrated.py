"""
integrated_app.py

Single-file Streamlit app:
- Restored original frontend (emojis, colors, sidebar, two-column layout, history)
- Improved backend: tone detection + 3 suggested replies (each labeled by tone)
- Uses google-generativeai (Gemini). Set GEMINI_API_KEY in environment or .env

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

# Load env vars
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
    created_at: datetime = Field(default_factory=datetime.now)

class Message(BaseModel):
    platform: Platform
    content: str
    sender: str
    priority: Priority = Priority.NORMAL
    timestamp: datetime = Field(default_factory=datetime.now)
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
    # Each suggested reply is a dict: {"tone": "<tone>", "text": "<reply text>"}
    suggested_replies: List[Dict[str,str]] = []
    reminders: List[Dict[str, Any]] = []

# ---------------------------
# AIAssistant
# ---------------------------
class AIAssistant:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the app.")
        genai.configure(api_key=api_key)
        # model choice as before; change if you prefer another model
        self.model = genai.GenerativeModel("gemini-2.5-flash")
        self.history: Dict[str, List[Dict[str, Any]]] = {}

    def _categorize_message(self, content: str) -> str:
        c = content.lower()
        if any(w in c for w in ["urgent", "asap", "emergency"]):
            return "urgent"
        if any(w in c for w in ["follow up", "follow-up", "followup"]):
            return "follow-up"
        if any(w in c for w in ["task:", "todo:", "to-do:"]):
            return "task"
        return "normal"

    def _extract_tasks(self, content: str, platform: Platform) -> List[Task]:
        tasks: List[Task] = []
        for line in content.splitlines():
            if "task:" in line.lower() or "todo:" in line.lower():
                title = line.split(":", 1)[1].strip()
                tasks.append(Task(
                    title=title,
                    description=title,
                    source_platform=platform
                ))
        return tasks

    def _generate_suggested_replies_with_tones(self, content: str) -> List[Dict[str,str]]:
        """
        Ask Gemini to return exactly three replies with explicit tone labels.
        We expect output in this format:
        1. Formal: <text>
        2. Friendly: <text>
        3. Neutral: <text>
        """
        try:
            prompt = f"""
You are an expert communications assistant.

A message was received:
\"\"\"{content}\"\"\"

Generate exactly three short reply options (each under 25 words) and label each with its tone.
Use the exact format below (three numbered lines):
1. Formal: <first reply>
2. Friendly: <second reply>
3. Neutral: <third reply>

Make replies professional, concise, and appropriate to replying to the provided message.
"""
            ai_resp = self.model.generate_content(prompt)
            text = getattr(ai_resp, "text", None) or str(ai_resp)

            # Try to extract "1. Tone: reply" patterns
            matches = re.findall(r"\d+\.\s*([A-Za-z\s/_-]+):\s*(.+)", text)
            replies: List[Dict[str,str]] = []
            for tone, reply in matches:
                t = tone.strip().lower()
                replies.append({"tone": t, "text": reply.strip().strip('"').strip("'")})

            # Fallback parsing if above didn't work well (split lines)
            if len(replies) < 3:
                lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
                parsed = []
                for ln in lines:
                    m = re.match(r"\d+\.\s*([A-Za-z\s/_-]+):\s*(.+)", ln)
                    if m:
                        parsed.append({"tone": m.group(1).strip().lower(), "text": m.group(2).strip()})
                    else:
                        # try "Formal - reply" or just take parts
                        m2 = re.match(r"\d+\.\s*([^:]+)[:-]\s*(.+)", ln)
                        if m2:
                            parsed.append({"tone": m2.group(1).strip().lower(), "text": m2.group(2).strip()})
                if len(parsed) >= 3:
                    replies = parsed[:3]

            # Ensure exactly 3 items; if missing, create safe defaults
            if len(replies) < 3:
                defaults = [
                    {"tone":"formal", "text":"Thank you for reaching out. I will review and get back to you shortly."},
                    {"tone":"friendly", "text":"Thanks! I’ll check this and follow up soon."},
                    {"tone":"neutral", "text":"Received. I will respond after reviewing."}
                ]
                # fill with what we have first
                for i in range(3):
                    if i < len(replies):
                        continue
                    replies.append(defaults[i])
            return replies[:3]
        except Exception:
            return [
                {"tone":"formal", "text":"Thank you for your message. I will review and respond shortly."},
                {"tone":"friendly", "text":"Thanks — I’ll take a look and get back to you soon."},
                {"tone":"neutral", "text":"Noted. I will follow up after reviewing."}
            ]

    async def process_message(self, message: Message) -> ResponseModel:
        # store history
        self.history.setdefault(message.sender, []).append({
            "content": message.content,
            "timestamp": message.timestamp
        })

        # categorize and tasks
        message.category = self._categorize_message(message.content)
        tasks = self._extract_tasks(message.content, message.platform)

        platform_prompt = {
            Platform.EMAIL: "You are an email assistant. Summarize and suggest replies for emails.",
            Platform.SLACK: "You are a Slack assistant. Summarize and identify action items in short.",
            Platform.WHATSAPP: "You are a WhatsApp assistant. Summarize messages and spot tone."
        }.get(message.platform, "")

        # Ask Gemini for summary + tone + one suggested response (for content field)
        context_prompt = f"""
{platform_prompt}

From: {message.sender}
Category: {message.category}
Priority: {message.priority}
Message:
\"\"\"{message.content}\"\"\"

Please provide:
- A brief one-line summary of the message.
- The dominant tone or emotion of the message (one word like: polite, urgent, angry, frustrated, formal, casual, neutral, friendly).
- Key action items (if any).
Return in plain text, e.g.:

Summary: <one-line summary>
Tone: <tone word>
Action items: <comma-separated items or 'none'>
"""

        ai_resp = self.model.generate_content(context_prompt)
        ai_text = getattr(ai_resp, "text", None) or str(ai_resp)

        # Parse summary, tone, action items with regex
        summary = None
        tone = None
        action_items: List[str] = []
        m_sum = re.search(r"Summary:\s*(.+)", ai_text, re.IGNORECASE)
        if m_sum:
            summary = m_sum.group(1).strip()
        m_tone = re.search(r"Tone:\s*(.+)", ai_text, re.IGNORECASE)
        if m_tone:
            tone = m_tone.group(1).strip()
        m_act = re.search(r"Action items?:\s*(.+)", ai_text, re.IGNORECASE)
        if m_act:
            act_text = m_act.group(1).strip()
            if act_text.lower() != "none":
                action_items = [a.strip() for a in re.split(r"[;,]", act_text) if a.strip()]

        # suggested replies with tone labels
        suggested = self._generate_suggested_replies_with_tones(message.content)

        reminders = []
        if message.category in ["urgent", "follow-up"]:
            reminders.append({
                "type": message.category,
                "due_date": datetime.now().isoformat(),
                "message": f"Follow up with {message.sender}: {message.content[:80]}..."
            })

        return ResponseModel(
            content=ai_text.strip(),
            summary=summary or f"Message from {message.sender}",
            tone=(tone or "not-specified"),
            action_items=action_items,
            tasks=tasks,
            priority=message.priority,
            suggested_replies=suggested,
            reminders=reminders
        )

assistant = AIAssistant()

def process_message_sync(message: Message) -> ResponseModel:
    # run coroutine in sync context
    return asyncio.run(assistant.process_message(message))

# ---------------------------
# Streamlit UI - restored layout + emojis + colors + copy button
# ---------------------------

st.set_page_config(page_title="AI Communication Assistant", page_icon="💬", layout="wide", initial_sidebar_state="expanded")

# CSS for message boxes and priority colors
st.markdown("""
    <style>
    .main { padding: 1rem; }
    .stButton>button { width: 100%; }
    .message-box { padding: 1rem; border-radius: 0.6rem; margin-bottom: 1rem; background-color: #f7f9fc; }
    .urgent { border-left: 5px solid #ff4b4b; }
    .high { border-left: 5px solid #ffa500; }
    .normal { border-left: 5px solid #00bfff; }
    .low { border-left: 5px solid #32cd32; }
    .follow_up { border-left: 5px solid #9370db; }
    .tone-badge { font-weight: 600; padding: 2px 8px; border-radius: 6px; background: #efefef; margin-right: 6px; }
    .tone-formal { background:#f0f3ff; }
    .tone-friendly { background:#e8f9f0; }
    .tone-neutral { background:#fff7e6; }
    </style>
""", unsafe_allow_html=True)

# session state init
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_message' not in st.session_state:
    st.session_state.current_message = None
if 'copied_text' not in st.session_state:
    st.session_state.copied_text = ""

st.title("💬 AI Communication Assistant")

# Sidebar with emojis and priority choices
with st.sidebar:
    st.header("Message Settings")
    platform_icons = {"email":"📧","slack":"💼","whatsapp":"📱"}
    platform = st.selectbox(
        "Platform",
        options=[Platform.EMAIL, Platform.SLACK, Platform.WHATSAPP],
        format_func=lambda x: f"{platform_icons.get(x.value,'')}  {x.value.capitalize()}"
    )

    priority_colors = {
        Priority.URGENT: "🔴 Urgent",
        Priority.HIGH: "🟠 High",
        Priority.NORMAL: "🟢 Normal",
        Priority.LOW: "⚪ Low",
        Priority.FOLLOW_UP: "🟣 Follow-up"
    }
    priority = st.selectbox(
        "Priority",
        options=[Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW, Priority.FOLLOW_UP],
        format_func=lambda x: priority_colors.get(x, x.value.capitalize())
    )

    sender = st.text_input("Sender", placeholder="Enter sender's email/ID or name")

    st.markdown("---")
    st.subheader("📊 Statistics")
    st.metric("Messages Processed", len(st.session_state.chat_history))
    active_tasks_count = sum(len(c['response'].get('tasks', [])) for c in st.session_state.chat_history)
    st.metric("Active Tasks", active_tasks_count)

# Main layout columns
col1, col2 = st.columns([3,4])

with col1:
    st.subheader("New Message")
    message_input = st.text_area("Type your message", placeholder="Type your message here...", height=150, help="Use 'Task:' to mark tasks.")
    send_disabled = not (message_input and sender)
    if st.button("📤 Send Message", type="primary", disabled=send_disabled):
        if not message_input or not sender:
            st.error("Please provide both message and sender.")
        else:
            with st.spinner("Processing message..."):
                try:
                    msg = Message(content=message_input, platform=platform, sender=sender, priority=priority)
                    resp = process_message_sync(msg)
                    payload = {
                        "content": resp.content,
                        "summary": resp.summary,
                        "tone": resp.tone,
                        "action_items": resp.action_items,
                        "tasks": [t.model_dump() for t in resp.tasks],
                        "priority": resp.priority.value,
                        "suggested_replies": resp.suggested_replies,
                        "reminders": resp.reminders
                    }
                    st.session_state.current_message = {"message": msg.model_dump(), "response": payload, "timestamp": datetime.now().isoformat()}
                    st.session_state.chat_history.append(st.session_state.current_message)
                    st.success("Message processed successfully!")
                except Exception as e:
                    st.error(f"Error processing message: {e}")

with col2:
    st.subheader("Response & Analysis")
    if st.session_state.current_message:
        r = st.session_state.current_message['response']
        # priority display with color class
        pr = r.get('priority','').lower()
        cls = "normal"
        if pr == "urgent": cls="urgent"
        elif pr == "high": cls="high"
        elif pr == "low": cls="low"
        elif pr == "follow_up": cls="follow_up"

        st.markdown(f"<div class='message-box {cls}'>", unsafe_allow_html=True)
        st.markdown(f"**Summary:** {r.get('summary','-')}")
        st.markdown(f"**Priority:** {r.get('priority','-').upper()}")
        st.markdown(f"**Tone:** {r.get('tone','-')}")
        # action items
        if r.get('action_items'):
            st.markdown("**Action Items:**")
            for ai in r['action_items']:
                st.markdown(f"- {ai}")
        # tasks
        if r.get('tasks'):
            st.markdown("**Extracted Tasks:**")
            for t in r['tasks']:
                st.markdown(f"- {t.get('title')} ({t.get('priority', '')})")
        # suggested replies with tone badges and copy button
        if r.get('suggested_replies'):
            st.markdown("#### 💬 Suggested Replies")
            for i, rep in enumerate(r['suggested_replies']):
                tone_label = rep.get('tone','').lower()
                tone_display = tone_label.capitalize()
                tone_class = "tone-badge"
                if "formal" in tone_label: tone_class += " tone-formal"
                elif "friend" in tone_label: tone_class += " tone-friendly"
                elif "neutral" in tone_label: tone_class += " tone-neutral"
                # show reply with badge and copy button
                cols = st.columns([8,2])
                with cols[0]:
                    st.markdown(f"<span class='{tone_class}'>{tone_display}</span> {rep.get('text','')}", unsafe_allow_html=True)
                with cols[1]:
                    # copy action - stores to session state and shows a tiny success
                    if st.button(f"Copy {i+1}", key=f"copy_{i}_{st.session_state.get('timestamp','') }"):
                        st.session_state.copied_text = rep.get('text','')
                        st.success("Copied to box below — use Ctrl+C to copy to clipboard")
            # small text area for copying convenience
            st.text_area("Copy area (click a Copy button to fill)", value=st.session_state.copied_text, height=80)

        # reminders
        if r.get('reminders'):
            st.markdown("#### ⏰ Reminders")
            for rem in r['reminders']:
                st.info(f"🔔 {rem.get('message','')} — Due: {rem.get('due_date','')}")
        st.markdown("</div>", unsafe_allow_html=True)

# Message history area (filters + display)
st.markdown("---")
st.subheader("📝 Message History")

col1f, col2f, col3f = st.columns(3)
with col1f:
    platform_filter = st.multiselect(
        "Filter by Platform",
        options=[p.value for p in Platform],
        default=[p.value for p in Platform]
    )
with col2f:
    priority_filter = st.multiselect(
        "Filter by Priority",
        options=[p.value for p in Priority],
        default=[p.value for p in Priority]
    )
with col3f:
    search = st.text_input("Search messages", placeholder="Type to search...")

for chat in reversed(st.session_state.chat_history):
    pm = chat['message']
    if (pm['platform'] in platform_filter and
        pm['priority'] in priority_filter and
        (not search or search.lower() in pm['content'].lower())):

        header = f"{pm['platform'].upper()} - {chat['timestamp']} - From: {pm['sender']}"
        with st.expander(header, expanded=False):
            st.markdown(f"**Original Message:**")
            st.markdown(f"```{pm['content']}```")
            resp = chat['response']
            if resp.get('tone'):
                st.markdown(f"**Tone:** {resp['tone']}")
            if resp.get('suggested_replies'):
                st.markdown("**Suggested Replies:**")
                for rep in resp['suggested_replies']:
                    st.markdown(f"- **{rep.get('tone', '').capitalize()}** — {rep.get('text','')}")
            if resp.get('tasks'):
                st.markdown("**Tasks:**")
                for t in resp['tasks']:
                    st.markdown(f"- {t.get('title')} ({t.get('priority','')})")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>AI Communication Assistant - Powered by Gemini</p>
        <p style='font-size: small'>Version 1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
