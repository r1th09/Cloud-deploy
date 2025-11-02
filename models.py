from enum import Enum
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from datetime import datetime

class Platform(str, Enum):
    """Enum for supported platforms"""
    EMAIL = "email"
    WHATSAPP = "whatsapp"
    SLACK = "slack"

class Priority(str, Enum):
    """Enum for message priorities"""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"
    FOLLOW_UP = "follow_up"

class TaskStatus(str, Enum):
    """Enum for task status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"

class Task(BaseModel):
    """Model for tasks extracted from messages"""
    title: str = Field(..., description="Task title", example="Complete section 3 of the documentation")
    description: str = Field(..., description="Task description", example="Complete section 3 of the project documentation by tomorrow")
    due_date: Optional[datetime] = Field(None, description="Task due date")
    priority: Priority = Field(default=Priority.NORMAL, description="Task priority level")
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    assignee: Optional[str] = Field(None, description="Task assignee email/ID", example="developer@company.com")
    source_platform: Platform = Field(..., description="Platform where task was created")
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Complete project documentation",
                "description": "Finish section 3 of the project documentation",
                "priority": "high",
                "status": "pending",
                "assignee": "developer@company.com",
                "source_platform": "email",
                "created_at": "2025-02-25T15:00:00"
            }
        }

class Message(BaseModel):
    """Message model for incoming messages"""
    platform: Platform = Field(..., description="Platform identifier (email, slack, whatsapp)")
    content: str = Field(..., description="Message content/body", example="Urgent: Need the project report by tomorrow")
    sender: str = Field(..., description="Message sender identifier", example="manager@company.com")
    priority: Priority = Field(default=Priority.NORMAL, description="Message priority level")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    category: Optional[str] = Field(None, description="Message category (urgent, follow-up, task)", example="urgent")
    requires_action: bool = Field(default=False, description="Whether message requires action")
    context: Optional[Dict[str, Any]] = Field(None, description="Thread/conversation context")

    class Config:
        json_schema_extra = {
            "example": {
                "platform": "email",
                "content": "Urgent: Need the project report by tomorrow. Task: Complete section 3 of the documentation.",
                "sender": "manager@company.com",
                "priority": "high",
                "category": "urgent",
                "requires_action": True,
                "timestamp": "2025-02-25T15:00:00"
            }
        }

class Response(BaseModel):
    """Response model for outgoing messages"""
    content: str = Field(..., description="AI-generated response content")
    summary: Optional[str] = Field(None, description="Brief summary of the message")
    action_items: List[str] = Field(default_factory=list, description="List of action items")
    tasks: List[Task] = Field(default_factory=list, description="List of extracted tasks")
    priority: Priority = Field(default=Priority.NORMAL, description="Response priority level")
    suggested_replies: List[str] = Field(default_factory=list, description="List of suggested quick replies")
    reminders: List[Dict[str, Any]] = Field(default_factory=list, description="List of generated reminders")
    context_updates: Optional[Dict[str, Any]] = Field(None, description="Updates to conversation context")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "I'll prioritize completing the project report and section 3 of the documentation by tomorrow.",
                "summary": "Urgent request for project report completion",
                "action_items": ["Complete project report", "Update documentation section 3"],
                "tasks": [{
                    "title": "Complete project documentation",
                    "description": "Finish section 3 of the project documentation",
                    "priority": "high",
                    "status": "pending",
                    "source_platform": "email"
                }],
                "priority": "urgent",
                "suggested_replies": [
                    "I'll complete this by tomorrow",
                    "Working on it right away",
                    "Will prioritize this task"
                ],
                "reminders": [{
                    "type": "urgent",
                    "due_date": "2025-02-26T15:00:00",
                    "message": "Follow up on project report"
                }]
            }
        }
