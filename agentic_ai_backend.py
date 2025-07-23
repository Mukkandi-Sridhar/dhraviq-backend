from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import os
import traceback
from dotenv import load_dotenv
from datetime import datetime
import firebase_admin
from firebase_admin import firestore, initialize_app, credentials
import google.generativeai as genai
import requests

# ------------------ Load Environment ------------------
load_dotenv()

# ------------------ Firebase Initialization ------------------
firebase_path = "firebase_credentials.json"

if not os.path.exists(firebase_path):
    raise FileNotFoundError(f"Firebase credentials not found at: {firebase_path}")

try:
    cred = credentials.Certificate(firebase_path)
    if not firebase_admin._apps:
        firebase_admin.initialize_app(cred)
    db = firestore.client()
except Exception as e:
    raise RuntimeError(f"Failed to initialize Firebase: {e}")

# ------------------ Gemini Model Setup ------------------
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# ------------------ Pushover Configuration ------------------
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
        "focus": "Helps users define clear, structured, and meaningful SMART goals.",
        "technical": "Breaks down ambitions into SMART components using goal-setting frameworks."
    },
    "SkillMap": {
        "name": "Skill Map",
        "focus": "Identifies skills required for success in a chosen goal or field.",
        "technical": "Maps out gaps and recommends courses, books, and learning paths."
    },
    "TimelineWizard": {
        "name": "Timeline Wizard",
        "focus": "Creates realistic timelines to achieve goals.",
        "technical": "Uses sprints, time-blocking, and backward planning with buffer periods."
    },
    "ProgressCoach": {
        "name": "Progress Coach",
        "focus": "Tracks execution and sustains consistency.",
        "technical": "Applies habit tracking, review loops, and adaptive feedback."
    },
    "MindsetMentor": {
        "name": "Mindset Mentor",
        "focus": "Builds discipline and mental resilience.",
        "technical": "Uses CBT tools, identity shifts, and habit reinforcement techniques."
    }
}

# ------------------ Pushover Notification ------------------
def send_pushover_notification(user_id: str, question: str, email: Optional[str] = None):
    try:
        message = f"New question from user {user_id}:\n\n{question}"
        if email:
            message += f"\n\nEmail: {email}"

        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "message": message,
                "title": "New User Question",
                "priority": 0,
                "sound": "magic",
                "html": 1
            },
            timeout=10
        )
        response.raise_for_status()
        return True
    except Exception as e:
        print(f"Pushover failed: {e}")
        return False

async def send_pushover_notification_async(user_id: str, question: str, email: Optional[str] = None):
    await asyncio.sleep(1)
    send_pushover_notification(user_id, question, email)

# ------------------ Keyword Extractor ------------------
def extract_tech_keywords(text: str) -> List[str]:
    tech_terms = [
        'python', 'javascript', 'react', 'node', 'django', 'flask',
        'machine learning', 'ai', 'data science', 'database',
        'frontend', 'backend', 'fullstack', 'devops'
    ]
    return [term for term in tech_terms if term in text.lower()]

# ------------------ Agent Response Generator ------------------
async def process_agent_response(agent: str, question: str) -> Dict:
    try:
        specialization = AGENT_SPECIALIZATIONS.get(agent, {
            "name": agent,
            "focus": "General guidance",
            "technical": "Provide thoughtful advice"
        })

        prompt = (
            f"You are an expert agent named {specialization['name']}.\n"
            f"Specialization: {specialization['focus']}\n"
            f"Technical Role: {specialization['technical']}\n\n"
            f"User Input: \"{question.strip()}\"\n\n"
            "Instructions:\n"
            "- If it's a greeting, give a short intro.\n"
            "- If it's a specific question, respond with detailed markdown advice.\n"
            "- Only focus on your role. No general answers.\n"
            "- Use markdown: **bold**, bullet points, numbered steps, etc.\n"
            "- Stay professional.\n\n"
            "Your Response:"
        )

        response = await asyncio.wait_for(
            asyncio.to_thread(gemini_model.generate_content, prompt, generation_config={
                "temperature": 0.3,
                "top_p": 0.95,
                "max_output_tokens": 2048
            }),
            timeout=12
        )

        return {
            "agent": agent,
            "response": response.text,
            "isTechnical": True
        }

    except Exception as e:
        print(f"[{agent}] Error: {e}")
        return {
            "agent": agent,
            "response": f"⚠️ {agent} is currently unavailable. Try again later.",
            "isTechnical": False
        }

# ------------------ Main Endpoint ------------------
@app.post("/run_agents")
async def run_agents(req: AgentRequest):
    try:
        if len(req.agents) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 agents allowed.")

        # Run all agents in parallel
        tasks = [process_agent_response(agent, req.question) for agent in req.agents]
        results = await asyncio.gather(*tasks)

        responses = {r["agent"]: r["response"] for r in results}

        # Store session
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

        # If reminder/email is set
        if req.send_email:
            db.collection("users").document(req.userId).set({
                "reminderEnabled": True,
                "reminderQuestion": req.question.strip(),
                "lastUpdated": firestore.SERVER_TIMESTAMP,
                "email": req.email if req.email else None
            }, merge=True)

            asyncio.create_task(send_pushover_notification_async(
                req.userId, req.question, req.email
            ))

        return {
            "status": "success",
            "sessionId": session_ref.id,
            "responses": responses
        }

    except Exception as e:
        error_id = datetime.utcnow().isoformat()
        print(f"[{error_id}] run_agents error:\n{traceback.format_exc()}")
        return {
            "status": "error",
            "message": "Something went wrong while processing your request.",
            "error_id": error_id
        }

# ------------------ Health Check ------------------
@app.get("/health")
async def health_check():
    try:
        db.collection("health").document("check").set({
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        pushover_ok = send_pushover_notification(
            "health-check", "Test from health endpoint", "health@example.com"
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
        raise HTTPException(500, detail=f"Health check failed: {str(e)}")

# ------------------ Run with Uvicorn ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
