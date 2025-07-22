from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import os
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, initialize_app, credentials
import google.generativeai as genai
import traceback
from datetime import datetime
import requests

# ------------------ Load Environment ------------------
load_dotenv()

# Firebase Initialization
cred = credentials.Certificate("dhraviq-firebase-adminsdk-fbsvc-00ee4536d0.json")
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()

# Gemini Model Setup
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Pushover Configuration
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Dhraviq AI Backend",
    version="5.1",
    description="Fast agentic backend with 5 agents and Pushover notifications"
)

# ------------------ Request Schema ------------------
class AgentRequest(BaseModel):
    userId: str
    question: str
    agents: List[str]
    email: Optional[str] = None
    send_email: Optional[bool] = False

# ------------------ Agent Roles ------------------
AGENT_SPECIALIZATIONS = {
    "GoalClarifier": {
        "name": "Goal Clarifier",
        "focus": "Helps users define clear, structured, and meaningful SMART goals aligned with their personal or professional direction.",
        "technical": "Breaks down vague ambitions into specific, measurable, achievable, relevant, and time-bound components using proven goal-setting methodologies."
    },
    "SkillMap": {
        "name": "Skill Map",
        "focus": "Assists users in identifying essential skills required to achieve their goals or succeed in a specific field.",
        "technical": "Maps out skill gaps and recommends personalized learning paths including online courses, books, certifications, and project-based practices."
    },
    "TimelineWizard": {
        "name": "Timeline Wizard",
        "focus": "Helps users plan realistic and efficient timelines for reaching their defined objectives.",
        "technical": "Structures work into milestones, sprints, or phases using backward planning, time-blocking, and Gantt-based strategies with buffer and review points."
    },
    "ProgressCoach": {
        "name": "Progress Coach",
        "focus": "Guides users in tracking progress, overcoming stagnation, and sustaining consistent execution over time.",
        "technical": "Uses behavior tracking, review loops, accountability systems, and adaptive iteration frameworks to maintain focus and course-correct effectively."
    },
    "MindsetMentor": {
        "name": "Mindset Mentor",
        "focus": "Supports users in building a growth-oriented, resilient, and disciplined mental framework aligned with long-term success.",
        "technical": "Applies cognitive-behavioral tools, habit-loop analysis, identity-shift models, and self-reflection techniques to strengthen motivation and mental clarity."
    }
}

# ------------------ Pushover Notification ------------------
def send_pushover_notification(user_id: str, question: str, email: Optional[str] = None):
    """Send notification to admin via Pushover"""
    try:
        message = f"New question from user {user_id}:\n\n{question}"
        if email:
            message += f"\n\nUser email: {email}"

        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "message": message,
                "title": "New User Question",
                "priority": 0,  # Normal priority
                "sound": "magic",  # Custom notification sound
                "html": 1  # Enable basic HTML formatting
            },
            timeout=10  # 10 second timeout
        )

        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        print(f"Pushover notification failed: {str(e)}")
        return False
    except Exception as e:
        print(f"Unexpected error sending Pushover notification: {str(e)}")
        return False

# ------------------ Agent Response Generator ------------------
async def process_agent_response(agent: str, question: str) -> Dict:
    """LLM decides if response should be a brief greeting or detailed help"""
    try:
        specialization = AGENT_SPECIALIZATIONS.get(agent, {
            "name": agent,
            "focus": "General guidance",
            "technical": "Provide thoughtful advice"
        })

        prompt = (
            f"You are an expert agent named {specialization['name']}.\n"
            f"Your role is to help users with the following specialization:\n"
            f"{specialization['focus']}\n\n"
            f"Technical expertise: {specialization['technical']}\n\n"
            f"User Input: \"{question.strip()}\"\n\n"
            "Instructions:\n"
            "- If the user input is a casual greeting (like 'hi', 'hello', 'hey'), reply with a short, formal one-line introduction stating your field of expertise.\n"
            "- If the user input is a topic-specific question related to your domain, give a detailed, insightful, and structured answer.\n"
            "- Focus strictly on your assigned specialization. Do not generalize or go beyond your expertise.\n"
            "- Use markdown formatting to enhance clarity:\n"
            "  • Use **bold** for important keywords or subheadings\n"
            "  • Use numbered steps or bullet points for methods, frameworks, or strategies\n"
            "  • Include relevant tools, techniques, or frameworks from your field\n"
            "- Never ask the user to rephrase their input. Respond constructively and helpfully regardless of input quality.\n"
            "- Maintain a professional, concise, and informative tone. Avoid casual or motivational language unless the context requires it.\n\n"
            "Your Response:"
        )

        response = gemini_model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 2048
            }
        )

        return {
            "agent": agent,
            "response": response.text,
            "isTechnical": True
        }

    except Exception as e:
        print(f"Agent {agent} error: {e}")
        return {
            "agent": agent,
            "response": f"⚠️ {agent} is currently unavailable. Please try again later.",
            "isTechnical": False
        }

# ------------------ Main Endpoint ------------------
@app.post("/run_agents")
async def run_agents(req: AgentRequest):
    try:
        if len(req.agents) > 5:
            raise HTTPException(400, "Maximum 5 agents allowed")

        # Step 1: Run agents in parallel
        tasks = [process_agent_response(agent, req.question) for agent in req.agents]
        results = await asyncio.gather(*tasks)
        responses = {r["agent"]: r["response"] for r in results}

        # Step 2: Store session as usual
        session_ref = db.collection("sessions").document()
        session_data = {
            "userId": req.userId,
            "question": req.question,
            "agents": req.agents,
            "responses": responses,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "isTechnical": any(r["isTechnical"] for r in results),
            "technicalKeywords": extract_tech_keywords(req.question)
        }
        session_ref.set(session_data)

        # Step 3: If reminders are enabled, save question and send notification
        if req.send_email:
            user_ref = db.collection("users").document(req.userId)
            user_ref.set({
                "reminderEnabled": True,
                "reminderQuestion": req.question.strip(),
                "lastUpdated": firestore.SERVER_TIMESTAMP,
                "email": req.email if req.email else None
            }, merge=True)

            print(f"✅ reminderQuestion saved: {req.question}")
            
            # Send Pushover notification in background
            asyncio.create_task(
                send_pushover_notification_async(
                    req.userId, 
                    req.question, 
                    req.email
                )
            )

        return {
            "status": "success",
            "sessionId": session_ref.id,
            "responses": responses
        }

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, detail=str(e))

# ------------------ Async Notification Helper ------------------
async def send_pushover_notification_async(user_id: str, question: str, email: Optional[str] = None):
    """Wrapper to run Pushover notification in background"""
    try:
        # Small delay to ensure main request completes first
        await asyncio.sleep(1)
        success = send_pushover_notification(user_id, question, email)
        if not success:
            print("Pushover notification failed after retry")
    except Exception as e:
        print(f"Async notification error: {str(e)}")

# ------------------ Keyword Extractor ------------------
def extract_tech_keywords(text: str) -> List[str]:
    tech_terms = [
        'python', 'javascript', 'react', 'node', 'django', 'flask',
        'machine learning', 'ai', 'data science', 'database',
        'frontend', 'backend', 'fullstack', 'devops'
    ]
    return [term for term in tech_terms if term in text.lower()]

# ------------------ Health Check ------------------
@app.get("/health")
async def health_check():
    try:
        db.collection("health").document("check").set({
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        
        # Test Pushover connectivity
        pushover_ok = send_pushover_notification(
            "health-check", 
            "This is a test notification from the health check endpoint",
            "health@example.com"
        )
        
        return {
            "status": "healthy",
            "services": {
                "firestore": "connected",
                "gemini": "available",
                "pushover": "connected" if pushover_ok else "unavailable"
            }
        }
    except Exception as e:
        raise HTTPException(500, detail=str(e))

# ------------------ App Entry Point ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
