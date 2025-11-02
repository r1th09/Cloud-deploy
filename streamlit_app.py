import streamlit as st
import requests
import json

st.title("AI Communication Assistant 💬")

st.write("Send a message to the AI assistant and see how it responds.")

content = st.text_area("Message content", "Urgent: Please review the proposal by EOD.")
platform = st.selectbox("Platform", ["email", "slack", "whatsapp"])
sender = st.text_input("Sender", "user@example.com")
priority = st.selectbox("Priority", ["normal", "high"])

if st.button("Send Message"):
    payload = {
        "content": content,
        "platform": platform,
        "sender": sender,
        "priority": priority
    }
    try:
        response = requests.post("http://localhost:8000/message", json=payload)
        if response.status_code == 200:
            data = response.json()
            st.subheader("AI Response")
            st.write(data["content"])
            st.subheader("Summary")
            st.write(data["summary"])
            st.subheader("Suggested Replies")
            st.write(data["suggested_replies"])
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    except Exception as e:
        st.error(f"Connection error: {e}")
