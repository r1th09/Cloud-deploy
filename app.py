import os
import uvicorn
from typing import Dict, Any, List
from fastapi import FastAPI, HTTPException
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from models import Message, Response, Platform, Priority, Task, TaskStatus

# Force reload environment variables
load_dotenv(override=True)

# Initialize FastAPI app with metadata
app = FastAPI(
    title="AI Communication Assistant",
    description="""
    An AI-powered assistant that helps manage messages across different platforms.
    
    Features:
    - Process messages from Email, Slack, and WhatsApp
    - Automatically categorize and prioritize messages
    - Extract tasks and create reminders
    - Generate AI-powered responses
    """,
    version="1.0.0"
)

class AIAssistant:
    def __init__(self):
        if not (api_key := os.getenv("GEMINI_API_KEY")):
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        print(f"Initializing with API key: {api_key}")  # Debug line
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.conversation_history: Dict[str, List[Dict[str, Any]]] = {}
    
    def _categorize_message(self, content: str, platform: Platform) -> str:
        """Categorize message based on content and platform"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["urgent", "asap", "emergency"]):
            return "urgent"
        elif any(word in content_lower for word in ["follow up", "followup", "follow-up"]):
            return "follow-up"
        elif any(word in content_lower for word in ["task", "todo", "to-do"]):
            return "task"
        return "normal"
    
    def _extract_tasks(self, content: str, platform: Platform) -> List[Task]:
        """Extract tasks from message content"""
        tasks = []
        # Simple task extraction based on keywords
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
        """Generate quick reply suggestions"""
        try:
            prompt = f"Generate 3 short, professional reply suggestions for this message: {content}"
            response = self.model.generate_content(prompt)
            suggestions = response.text.split("\n")
            return [s.strip() for s in suggestions if s.strip()]
        except Exception:
            return ["Thank you for your message.", "I'll look into this.", "Let me get back to you."]
    
    async def process_message(self, message: Message) -> Response:
        """Process incoming message and generate response"""
        try:
            # Update conversation history
            if message.sender not in self.conversation_history:
                self.conversation_history[message.sender] = []
            self.conversation_history[message.sender].append({
                "content": message.content,
                "timestamp": message.timestamp
            })
            
            # Categorize message
            message.category = self._categorize_message(message.content, message.platform)
            
            # Extract tasks
            tasks = self._extract_tasks(message.content, message.platform)
            
            # Prepare system prompt based on platform
            platform_prompts = {
                Platform.EMAIL: "You are an email assistant. Help prioritize and suggest responses for emails.",
                Platform.SLACK: "You are a Slack assistant. Help summarize conversations and identify action items.",
                Platform.WHATSAPP: "You are a WhatsApp assistant. Help manage personal and business communications."
            }
            
            system_prompt = platform_prompts.get(message.platform)
            
            # Generate context for AI
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
            
            # Generate response using AI
            ai_response = self.model.generate_content(context)
            
            # Generate quick reply suggestions
            suggested_replies = self._generate_suggested_replies(message.content)
            
            # Create reminders if needed
            reminders = []
            if message.category in ["urgent", "follow-up"]:
                reminders.append({
                    "type": message.category,
                    "due_date": datetime.now(),  # You would calculate appropriate due date
                    "message": f"Follow up with {message.sender} regarding: {message.content[:100]}..."
                })
            
            return Response(
                content=ai_response.text,
                summary=f"Message from {message.sender} - {message.category.upper()}",
                action_items=[f"Reply to {message.sender}"] if message.category == "urgent" else [],
                tasks=tasks,
                priority=Priority.URGENT if message.category == "urgent" else message.priority,
                suggested_replies=suggested_replies,
                reminders=reminders
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

# Initialize AI Assistant
assistant = AIAssistant()



@app.post("/message", 
    response_model=Response,
    summary="Process a new message",
    description="""
    Process a new message from any supported platform (email, slack, whatsapp).
    
    The system will:
    - Categorize the message priority
    - Extract any tasks
    - Generate an AI response
    - Create reminders if needed
    
    Example message:
    ```json
    {
        "content": "Urgent: Need the project report by tomorrow. Task: Complete section 3.",
        "platform": "email",
        "sender": "manager@example.com",
        "priority": "normal"
    }
    ```
    """
)
async def handle_message(message: Message) -> Response:
    """Handle incoming messages from any platform"""
    try:
        return await assistant.process_message(message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health",
    summary="Health check endpoint",
    description="Check if the service is running and healthy"
)
def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now()}

if __name__ == "__main__":

    uvicorn.run(
        "app:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("DEBUG_MODE", "false").lower() == "true"
    )