from typing import List, Optional, Dict
from datetime import datetime
import os
import asyncio
import traceback
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, credentials
import google.generativeai as genai
import requests
from pydantic import BaseModel

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

# ------------------ Pydantic Request Schema ------------------
class AgentRequest(BaseModel):
    userId: str
    question: str
    agents: List[str]
    email: Optional[str] = None
    send_email: Optional[bool] = False

# ------------------ Notification ------------------
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
            "response": getattr(response, "text", "⚠️ No response returned."),
            "isTechnical": True
        }

    except Exception as e:
        print(f"[{agent}] Error: {e}")
        return {
            "agent": agent,
            "response": f"⚠️ {agent} is currently unavailable. Try again later.",
            "isTechnical": False
        }

# ------------------ Main Logic Function ------------------
async def run_agentic_logic(req: AgentRequest) -> Dict:
    try:
        if len(req.agents) > 5:
            raise ValueError("Maximum 5 agents allowed.")

        tasks = [process_agent_response(agent, req.question) for agent in req.agents]
        results = await asyncio.gather(*tasks)
        responses = {r["agent"]: str(r["response"]) for r in results]

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

        try:
            session_ref.set(session_data)
        except Exception:
            print(f"[{datetime.utcnow().isoformat()}] Firebase Error: {traceback.format_exc()}")

        if req.send_email:
            try:
                db.collection("users").document(req.userId).set({
                    "reminderEnabled": True,
                    "reminderQuestion": req.question.strip(),
                    "lastUpdated": firestore.SERVER_TIMESTAMP,
                    "email": req.email if req.email else None
                }, merge=True)

                asyncio.create_task(send_pushover_notification_async(
                    req.userId, req.question, req.email
                ))
            except Exception:
                print(f"[{datetime.utcnow().isoformat()}] Pushover Error: {traceback.format_exc()}")

        return {
            "status": "success",
            "sessionId": session_ref.id,
            "responses": responses
        }

    except Exception as e:
        error_id = datetime.utcnow().isoformat()
        print(f"[{error_id}] run_agentic_logic error:\n{traceback.format_exc()}")
        return {
            "status": "success",
            "responses": {
                "System": f"⚠️ Something went wrong. Please try again later. (Error ID: {error_id})"
            }
        }
