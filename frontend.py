import streamlit as st
import requests
import json
from datetime import datetime
from models import Platform, Priority

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
        format_func=lambda x: f"{platform_icons.get(x, '')} {x.capitalize()}"
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
        format_func=lambda x: priority_colors.get(x, x.capitalize())
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
                    # Prepare request
                    payload = {
                        "content": message,
                        "platform": platform,
                        "sender": sender,
                        "priority": priority,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Send to backend
                    response = requests.post(
                        "http://localhost:8000/message",
                        json=payload
                    )
                    response.raise_for_status()
                    
                    # Process response
                    data = response.json()
                    st.session_state.current_message = {
                        "message": payload,
                        "response": data,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.chat_history.append(st.session_state.current_message)
                    st.success("Message processed successfully!")
                    
                except requests.exceptions.RequestException as e:
                    st.error(f"Error: {str(e)}")

with col2:
    st.subheader("Response & Analysis")
    
    # Show current message response
    if st.session_state.current_message:
        with st.container():
            response = st.session_state.current_message['response']
            
            # Summary and priority
            st.markdown(f"**Priority:** {response['priority'].upper()}")
            if response.get('summary'):
                st.markdown(f"**Summary:** {response['summary']}")
            
            # Tasks
            if response.get('tasks'):
                st.markdown("#### 📋 Tasks")
                for task in response['tasks']:
                    with st.expander(f"Task: {task['title']}", expanded=True):
                        st.markdown(f"**Description:** {task['description']}")
                        st.markdown(f"**Priority:** {task['priority']}")
                        st.markdown(f"**Status:** {task['status']}")
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

# Message history
st.markdown("---")
st.subheader("📝 Message History")

# Filters for history
col1, col2, col3 = st.columns(3)
with col1:
    platform_filter = st.multiselect(
        "Filter by Platform",
        options=[p.value for p in Platform],
        default=[p.value for p in Platform]
    )
with col2:
    priority_filter = st.multiselect(
        "Filter by Priority",
        options=[p.value for p in Priority],
        default=[p.value for p in Priority]
    )
with col3:
    search = st.text_input("Search messages", placeholder="Type to search...")

# Display filtered history
for chat in reversed(st.session_state.chat_history):
    if (chat['message']['platform'] in platform_filter and
        chat['message']['priority'] in priority_filter and
        (not search or search.lower() in chat['message']['content'].lower())):
        
        with st.expander(
            f"{chat['message']['platform'].upper()} - {chat['timestamp']} - From: {chat['message']['sender']}",
            expanded=False
        ):
            # Original message
            st.markdown(f"**Original Message:**")
            st.markdown(f"```{chat['message']['content']}```")
            
            # Response details
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Platform:** {chat['message']['platform']}")
            with col2:
                st.markdown(f"**Priority:** {chat['message']['priority']}")
            
            if chat['response'].get('tasks'):
                st.markdown("**Tasks:**")
                for task in chat['response']['tasks']:
                    st.markdown(f"- {task['title']} ({task['priority']})")
            
            if chat['response'].get('suggested_replies'):
                st.markdown("**Suggested Replies:**")
                for reply in chat['response']['suggested_replies']:
                    st.markdown(f"```\n{reply}\n```")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>AI Communication Assistant - Powered by Gemini Pro</p>
        <p style='font-size: small'>Version 1.0.0</p>
    </div>
    """,
    unsafe_allow_html=True
)
