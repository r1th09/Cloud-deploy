"""
integrated_app.py

Single-file Streamlit app that contains:
- The Streamlit frontend (UI) from frontend.py
- The FastAPI backend logic (AIAssistant) from app.py, adapted to run synchronously inside Streamlit
- Pydantic models inlined from models.py

Run:
    streamlit run integrated_app.py

Requirements (from your requirements.txt):
    streamlit, google-generativeai, python-dotenv, pydantic

Important:
- Set GEMINI_API_KEY in environment before running (Streamlit Cloud -> Secrets / Environment variables).
"""

import os
import streamlit as st
from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum
import asyncio
from dotenv import load_dotenv

# AI SDK
import google.generativeai as genai

# Load env vars
load_dotenv(override=True)

# ---------------------------
# Models (inlined from models.py)
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
    title: str = Field(..., description="Task title")
    description: str = Field(..., description="Task description")
    due_date: Optional[datetime] = Field(None, description="Task due date")
    priority: Priority = Field(default=Priority.NORMAL, description="Task priority level")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    assignee: Optional[str] = Field(None, description="Task assignee email/ID")
    source_platform: Platform = Field(..., description="Platform where task was created")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")

class Message(BaseModel):
    platform: Platform = Field(..., description="Platform identifier (email, slack, whatsapp)")
    content: str = Field(..., description="Message content/body")
    sender: str = Field(..., description="Message sender identifier")
    priority: Priority = Field(default=Priority.NORMAL, description="Message priority level")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    category: Optional[str] = Field(None, description="Message category (urgent, follow-up, task)")
    requires_action: bool = Field(default=False, description="Whether message requires action")
    context: Optional[Dict[str, Any]] = Field(None, description="Thread/conversation context")

class ResponseModel(BaseModel):
    content: str = Field(..., description="AI-generated response content")
    summary: Optional[str] = Field(None, description="Brief summary of the message")
    action_items: List[str] = Field(default_factory=list, description="List of action items")
    tasks: List[Task] = Field(default_factory=list, description="List of extracted tasks")
    priority: Priority = Field(default=Priority.NORMAL, description="Response priority level")
    suggested_replies: List[str] = Field(default_factory=list, description="List of suggested quick replies")
    reminders: List[Dict[str, Any]] = Field(default_factory=list, description="List of generated reminders")
    context_updates: Optional[Dict[str, Any]] = Field(None, description="Updates to conversation context")


# ---------------------------
# AIAssistant (adapted from app.py)
# ---------------------------
class AIAssistant:
    def __init__(self):
        if not (api_key := os.getenv("GEMINI_API_KEY")):
            raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the app.")
        # configure Gemini
        genai.configure(api_key=api_key)
        # choose model - keep same as in your original app
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}

    def _categorize_message(self, content: str, platform: Platform) -> str:
        content_lower = content.lower()
        if any(word in content_lower for word in ["urgent", "asap", "emergency"]):
            return "urgent"
        elif any(word in content_lower for word in ["follow up", "followup", "follow-up"]):
            return "follow-up"
        elif any(word in content_lower for word in ["task", "todo", "to-do"]):
            return "task"
        return "normal"

    def _extract_tasks(self, content: str, platform: Platform) -> List[Task]:
        tasks: List[Task] = []
        if "task:" in content.lower() or "todo:" in content.lower():
            lines = content.split("\n")
            for line in lines:
                if "task:" in line.lower() or "todo:" in line.lower():
                    task_content = line.split(":", 1)[1].strip()
                    tasks.append(Task(
                        title=task_content,
                        description=task_content,
                        source_platform=platform,
                        priority=Priority.NORMAL
                    ))
        return tasks

    def _generate_suggested_replies(self, content: str) -> List[str]:
        try:
            prompt = f"Generate 3 short, professional reply suggestions for this message: {content}"
            # Use the model to generate short suggestions
            # Many Gemini SDKs return an object; we follow same pattern as original code.
            response = self.model.generate_content(prompt)
            # Gemini response may include text or structured fields; try to read `.text`
            suggestions_text = getattr(response, "text", None) or str(response)
            # split lines and clean
            suggestions = [s.strip() for s in suggestions_text.split("\n") if s.strip()]
            # ensure at least 3 items
            if len(suggestions) >= 3:
                return suggestions[:3]
            # fallback if single string with commas
            if len(suggestions) == 1 and "," in suggestions[0]:
                parts = [p.strip() for p in suggestions[0].split(",") if p.strip()]
                return parts[:3]
            return suggestions or ["Thank you for your message.", "I'll look into this.", "Let me get back to you."]
        except Exception as e:
            # Non-blocking fallback
            return ["Thank you for your message.", "I'll look into this.", "Let me get back to you."]

    async def process_message(self, message: Message) -> ResponseModel:
        try:
            # conversation history
            if message.sender not in self.conversation_history:
                self.conversation_history[message.sender] = []
            self.conversation_history[message.sender].append({
                "content": message.content,
                "timestamp": message.timestamp
            })

            # categorize and extract tasks
            message.category = self._categorize_message(message.content, message.platform)
            tasks = self._extract_tasks(message.content, message.platform)

            # craft system prompt per platform
            platform_prompts = {
                Platform.EMAIL: "You are an email assistant. Help prioritize and suggest responses for emails.",
                Platform.SLACK: "You are a Slack assistant. Help summarize conversations and identify action items.",
                Platform.WHATSAPP: "You are a WhatsApp assistant. Help manage personal and business communications."
            }
            system_prompt = platform_prompts.get(message.platform, "")

            # Prepare full context for the model
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
            # Call the Gemini model to generate content
            ai_response = self.model.generate_content(context)
            ai_text = getattr(ai_response, "text", None) or str(ai_response)

            suggested_replies = self._generate_suggested_replies(message.content)

            reminders = []
            if message.category in ["urgent", "follow-up"]:
                reminders.append({
                    "type": message.category,
                    "due_date": datetime.now().isoformat(),
                    "message": f"Follow up with {message.sender} regarding: {message.content[:100]}."
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
        except Exception as e:
            # re-raise so UI can show error
            raise e

# instantiate assistant (will raise if GEMINI_API_KEY missing)
assistant = AIAssistant()


# Helper: sync wrapper for the async process_message
def process_message_sync(message: Message) -> ResponseModel:
    """
    Run the assistant.process_message coroutine synchronously.
    Uses asyncio.run() which is safe here because Streamlit runs in a sync context.
    """
    return asyncio.run(assistant.process_message(message))


# ---------------------------
# Streamlit UI (from frontend.py) - kept the same interface and look
# ---------------------------

# Page configuration with custom theme
st.set_page_config(
    page_title="AI Communication Assistant",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
    }
    .message-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        background-color: #f0f2f6;
    }
    .urgent {
        border-left: 4px solid #ff4b4b;
    }
    .normal {
        border-left: 4px solid #00bfff;
    }
    .high {
        border-left: 4px solid #ffa500;
    }
    .low {
        border-left: 4px solid #32cd32;
    }
    .follow_up {
        border-left: 4px solid #9370db;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_message' not in st.session_state:
    st.session_state.current_message = None

# Main layout
st.title("💬 AI Communication Assistant")

# Sidebar configuration
with st.sidebar:
    st.header("Message Settings")
    
    # Platform selection with icons
    platform_icons = {
        "email": "📧",
        "slack": "💼",
        "whatsapp": "📱"
    }
    platform = st.selectbox(
        "Platform",
        options=[Platform.EMAIL, Platform.SLACK, Platform.WHATSAPP],
        format_func=lambda x: f"{platform_icons.get(x.value, '')} {x.value.capitalize()}"
    )
    
    # Priority selection with color indicators
    priority_colors = {
        "urgent": "🔴 Urgent",
        "high": "🟠 High",
        "normal": "🟢 Normal",
        "low": "⚪ Low",
        "follow_up": "🟣 Follow-up"
    }
    priority = st.selectbox(
        "Priority",
        options=[Priority.URGENT, Priority.HIGH, Priority.NORMAL, Priority.LOW, Priority.FOLLOW_UP],
        format_func=lambda x: priority_colors.get(x.value, x.value.capitalize())
    )
    
    # Sender input with validation
    sender = st.text_input(
        "Sender",
        placeholder="Enter sender's email/ID",
        help="Email address for email, username for Slack/WhatsApp"
    )
    
    st.markdown("---")
    
    # Stats and info
    st.subheader("📊 Statistics")
    st.metric("Messages Processed", len(st.session_state.chat_history))
    st.metric("Active Tasks", sum(1 for chat in st.session_state.chat_history 
                                if chat.get('response', {}).get('tasks', [])))

# Main content area - Two columns
col1, col2 = st.columns([3, 4])

with col1:
    st.subheader("New Message")
    
    # Message input
    message = st.text_area(
        "Type your message",
        placeholder="Type your message here...",
        height=150,
        help="Enter your message content. Use 'Task:' prefix to mark tasks."
    )
    
    # Send button with validation
    if st.button("📤 Send Message", type="primary", disabled=not (message and sender)):
        if not message or not sender:
            st.error("Please provide both message and sender information.")
        else:
            with st.spinner("Processing message..."):
                try:
                    # Build Message model
                    msg = Message(
                        content=message,
                        platform=platform,
                        sender=sender,
                        priority=priority,
                        timestamp=datetime.now()
                    )
                    # Process message using assistant directly (no HTTP)
                    resp = process_message_sync(msg)

                    # Prepare payload in same shape the frontend previously expected
                    payload = {
                        "content": resp.content,
                        "summary": resp.summary,
                        "action_items": resp.action_items,
                        "tasks": [t.dict() for t in resp.tasks],
                        "priority": resp.priority.value,
                        "suggested_replies": resp.suggested_replies,
                        "reminders": resp.reminders
                    }

                    st.session_state.current_message = {
                        "message": msg.dict(),
                        "response": payload,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(st.session_state.current_message)
                    st.success("Message processed successfully!")
                except Exception as e:
                    st.error(f"Error processing message: {e}")

with col2:
    st.subheader("Response & Analysis")
    
    # Show current message response
    if st.session_state.current_message:
        with st.container():
            response = st.session_state.current_message['response']
            
            # Summary and priority
            st.markdown(f"**Priority:** {response.get('priority', 'N/A').upper()}")
            if response.get('summary'):
                st.markdown(f"**Summary:** {response['summary']}")
            
            # Tasks
            if response.get('tasks'):
                st.markdown("#### 📋 Tasks")
                for task in response['tasks']:
                    with st.expander(f"Task: {task['title']}", expanded=True):
                        st.markdown(f"**Description:** {task.get('description', '')}")
                        st.markdown(f"**Priority:** {task.get('priority', '')}")
                        st.markdown(f"**Status:** {task.get('status', '')}")
                        if task.get('due_date'):
                            st.markdown(f"**Due:** {task['due_date']}")
            
            # Suggested replies
            if response.get('suggested_replies'):
                st.markdown("#### 💬 Suggested Replies")
                for reply in response['suggested_replies']:
                    st.code(reply, language=None)
            
            # Reminders
            if response.get('reminders'):
                st.markdown("#### ⏰ Reminders")
                for reminder in response['reminders']:
                    st.info(f"🔔 {reminder['message']}\nDue: {reminder['due_date']}")

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

# Display filtered history
for chat in reversed(st.session_state.chat_history):
    pm = chat['message']
    if (pm['platform'] in platform_filter and
        pm['priority'] in priority_filter and
        (not search or search.lower() in pm['content'].lower())):
        
        with st.expander(
            f"{pm['platform'].upper()} - {chat['timestamp']} - From: {pm['sender']}",
            expanded=False
        ):
            st.markdown(f"**Original Message:**")
            st.markdown(f"```{pm['content']}```")
            
            colA, colB = st.columns(2)
            with colA:
                st.markdown(f"**Platform:** {pm['platform']}")
            with colB:
                st.markdown(f"**Priority:** {pm['priority']}")
            
            resp = chat['response']
            if resp.get('tasks'):
                st.markdown("**Tasks:**")
                for t in resp['tasks']:
                    st.markdown(f"- {t['title']} ({t.get('priority', '')})")
            
            if resp.get('suggested_replies'):
                st.markdown("**Suggested Replies:**")
                for reply in resp['suggested_replies']:
                    st.markdown(f"```\n{reply}\n```")

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
