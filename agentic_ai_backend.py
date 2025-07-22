from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import os
import json
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, initialize_app, credentials
import google.generativeai as genai
import traceback
from datetime import datetime
import requests
import logging

# ------------------ Load Environment Variables ------------------
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

# Global variables for initialized clients
db = None
gemini_model = None
PUSHOVER_TOKEN = None
PUSHOVER_USER = None

# ------------------ Firebase Initialization ------------------
# Load from Render secret or local file
secret_path = "/run/secrets/firebase-service-account.json"
if os.path.exists(secret_path):
    logging.debug("Loading Firebase credentials from Render secret.")
    with open(secret_path, 'r') as f:
        cred_dict = json.load(f)
else:
    local_path = "firebase-service-account.json"
    logging.debug(f"Loading Firebase credentials from local file: {local_path}")
    with open(local_path, 'r') as f:
        cred_dict = json.load(f)

try:
    cred = credentials.Certificate(cred_dict)
    if not firebase_admin._apps:
        initialize_app(cred)
    db = firestore.client()
    logging.info("Firebase initialized successfully.")
except Exception as e:
    logging.error(f"Firebase initialization failed: {str(e)}")
    db = None  # Set to None but allow app to proceed with error response

# ------------------ Gemini Model Setup ------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
logging.debug(f"GEMINI_API_KEY loaded: {GEMINI_API_KEY is not None}")
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_model = genai.GenerativeModel("gemini-1.5-pro")
        test_response = gemini_model.generate_content("Test")
        logging.debug(f"Gemini test response: {test_response.text}")
        logging.info("Gemini model configured successfully.")
    except Exception as e:
        logging.error(f"Gemini initialization failed: {str(e)}")
        gemini_model = None
else:
    logging.warning("GEMINI_API_KEY not found. Gemini will be unavailable.")

# ------------------ Pushover Configuration ------------------
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
logging.debug(f"PUSHOVER_TOKEN loaded: {PUSHOVER_TOKEN is not None}, PUSHOVER_USER loaded: {PUSHOVER_USER is not None}")
if not PUSHOVER_TOKEN or not PUSHOVER_USER:
    logging.warning("Pushover credentials not set. Notifications will not work.")

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
        "technical": "Breaks down goals into SMART components."
    },
    "SkillMap": {
        "name": "Skill Map",
        "focus": "Identifies skills needed for user goals.",
        "technical": "Recommends learning paths and resources."
    },
    "TimelineWizard": {
        "name": "Timeline Wizard",
        "focus": "Plans realistic timelines for objectives.",
        "technical": "Uses backward planning and Gantt strategies."
    },
    "ProgressCoach": {
        "name": "Progress Coach",
        "focus": "Tracks progress and overcomes stagnation.",
        "technical": "Implements behavior tracking and review loops."
    },
    "MindsetMentor": {
        "name": "Mindset Mentor",
        "focus": "Builds a growth-oriented mindset.",
        "technical": "Applies cognitive-behavioral tools."
    }
}

# ------------------ Pushover Notification ------------------
def send_pushover_notification(user_id: str, question: str, email: Optional[str] = None) -> bool:
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        logging.warning("Pushover credentials missing. Skipping notification.")
        return False

    try:
        message = f"New question from {user_id}:\n{question}"
        if email:
            message += f"\nEmail: {email}"
        response = requests.post(
            "https://api.pushover.net/1/messages.json",
            data={
                "token": PUSHOVER_TOKEN,
                "user": PUSHOVER_USER,
                "message": message,
                "title": "New User Question",
                "priority": 0,
                "sound": "magic"
            },
            timeout=10
        )
        response.raise_for_status()
        logging.info("Pushover notification sent successfully.")
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Pushover notification failed: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected Pushover error: {str(e)}")
        return False

async def send_pushover_notification_async(user_id: str, question: str, email: Optional[str] = None):
    await asyncio.to_thread(send_pushover_notification, user_id, question, email)

# ------------------ Agent Response Generator ------------------
async def process_agent_response(agent: str, question: str) -> Dict:
    if gemini_model is None:
        error_msg = f"⚠️ {agent} unavailable: Gemini model not initialized."
        logging.error(error_msg)
        return {"agent": agent, "response": error_msg, "isTechnical": False}

    try:
        specialization = AGENT_SPECIALIZATIONS.get(agent, {
            "name": agent, "focus": "General guidance", "technical": "Provide advice"
        })
        prompt = (
            f"You are {specialization['name']}.\nFocus: {specialization['focus']}\n"
            f"Technical: {specialization['technical']}\n\nUser: \"{question}\"\n\n"
            "Respond with a brief greeting for casual input (e.g., 'hi'), or a detailed answer for questions. Use markdown (**bold**, bullets) and stay within your expertise."
        )
        response = await asyncio.to_thread(
            lambda: gemini_model.generate_content(prompt).text
        )
        return {"agent": agent, "response": response, "isTechnical": True}
    except Exception as e:
        error_msg = f"⚠️ {agent} error: {str(e)}. Try again later."
        logging.error(f"Agent {agent} error: {str(e)}")
        logging.debug(traceback.format_exc())
        return {"agent": agent, "response": error_msg, "isTechnical": False}

# ------------------ Main Endpoint Logic ------------------
async def run_agents(req: AgentRequest):
    try:
        # Check dependencies
        if db is None:
            error_msg = "⚠️ Service unavailable: Database connection failed."
            logging.error(error_msg)
            return {"status": "error", "message": error_msg, "sessionId": None, "responses": {}}
        if gemini_model is None:
            error_msg = "⚠️ Service unavailable: AI model initialization failed."
            logging.error(error_msg)
            return {"status": "error", "message": error_msg, "sessionId": None, "responses": {}}

        if len(req.agents) > 5:
            error_msg = "⚠️ Maximum 5 agents allowed."
            logging.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)

        # Run agents in parallel
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
            "isTechnical": any(r["isTechnical"] for r in results)
        }
        session_ref.set(session_data)
        logging.debug(f"Session saved: {session_ref.id}")

        # Handle reminders
        if req.send_email:
            user_ref = db.collection("users").document(req.userId)
            user_ref.set({
                "reminderEnabled": True,
                "reminderQuestion": req.question.strip(),
                "lastUpdated": firestore.SERVER_TIMESTAMP,
                "email": req.email
            }, merge=True)
            logging.info(f"Reminder saved for {req.userId}")
            asyncio.create_task(send_pushover_notification_async(req.userId, req.question, req.email))

        return {"status": "success", "sessionId": session_ref.id, "responses": responses}

    except HTTPException as e:
        logging.error(f"HTTP error: {str(e)}")
        return {"status": "error", "message": str(e.detail), "sessionId": None, "responses": {}}
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        logging.debug(traceback.format_exc())
        return {"status": "error", "message": f"⚠️ Unexpected error: {str(e)}. Please try again.", "sessionId": None, "responses": {}}

# ------------------ Keyword Extractor ------------------
def extract_tech_keywords(text: str) -> List[str]:
    tech_terms = ['python', 'javascript', 'react', 'node', 'django', 'flask', 'machine learning', 'ai']
    return [term for term in tech_terms if term in text.lower()]

# ------------------ Health Check ------------------
async def health_check():
    health = {"firestore": "unavailable", "gemini": "unavailable", "pushover": "unavailable"}
    if db:
        try:
            db.collection("health_checks").document("api_status").set({"last_checked": firestore.SERVER_TIMESTAMP})
            health["firestore"] = "connected"
        except Exception as e:
            health["firestore"] = f"error: {str(e)}"
    if gemini_model:
        try:
            response = await asyncio.to_thread(lambda: gemini_model.generate_content("health check").text)
            health["gemini"] = "available" if response else "error"
        except Exception as e:
            health["gemini"] = f"error: {str(e)}"
    pushover_ok = send_pushover_notification("health-check", "Health check")
    health["pushover"] = "connected" if pushover_ok else "unavailable"
    return {
        "status": "healthy" if all(status in ["connected", "available"] for status in health.values()) else "unhealthy",
        "services": health,
        "timestamp": datetime.now().isoformat()
    }

# ------------------ FastAPI App ------------------
app = FastAPI(title="Dhraviq Agentic AI Backend", description="Multi-agent AI", version="1.0.0")

@app.get("/health")
async def health_endpoint():
    return await health_check()

@app.post("/run_agents")
async def run_agents_endpoint(req: AgentRequest):
    return await run_agents(req)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
