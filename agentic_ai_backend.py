from fastapi import HTTPException
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
import time

# ------------------ Load Environment Variables ------------------
load_dotenv()
logging.basicConfig(level=logging.DEBUG)

# Global variables for initialized clients
db = None
gemini_model = None
PUSHOVER_TOKEN = None
PUSHOVER_USER = None

# ------------------ Firebase Initialization ------------------
FIREBASE_SERVICE_ACCOUNT_KEY_JSON = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')
logging.debug(f"FIREBASE_SERVICE_ACCOUNT_KEY_JSON loaded: {FIREBASE_SERVICE_ACCOUNT_KEY_JSON is not None}")

# Initialize Firebase client
if FIREBASE_SERVICE_ACCOUNT_KEY_JSON:
    try:
        cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_KEY_JSON)
        cred = credentials.Certificate(cred_dict)
        if not firebase_admin._apps:
            initialize_app(cred)
        db = firestore.client()
        logging.info("Firebase Admin SDK initialized successfully from environment variable.")
        logging.debug("Firebase app initialized with credentials.")
    except json.JSONDecodeError as e:
        logging.error(f"CRITICAL ERROR: Error decoding FIREBASE_SERVICE_ACCOUNT_KEY JSON: {e}")
        db = None  # Set db to None to indicate initialization failure
    except Exception as e:
        logging.error(f"CRITICAL ERROR: Error during Firebase Admin SDK initialization: {e}")
        db = None  # Set db to None to indicate initialization failure
else:
    logging.warning("FIREBASE_SERVICE_ACCOUNT_KEY environment variable not found. Firestore operations will be unavailable.")

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
        logging.error(f"CRITICAL ERROR: Error configuring Gemini model: {e}")
        gemini_model = None  # Set gemini_model to None if initialization fails
else:
    logging.warning("GEMINI_API_KEY environment variable not found. Gemini will be unavailable.")

# ------------------ Pushover Configuration ------------------
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
logging.debug(f"PUSHOVER_TOKEN loaded: {PUSHOVER_TOKEN is not None}, PUSHOVER_USER loaded: {PUSHOVER_USER is not None}")
if not PUSHOVER_TOKEN or not PUSHOVER_USER:
    logging.warning("WARNING: Pushover API keys (PUSHOVER_TOKEN, PUSHOVER_USER) not found. Notifications will not work.")

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
def send_pushover_notification(user_id: str, question: str, email: Optional[str] = None) -> bool:
    """Send notification to admin via Pushover"""
    if not PUSHOVER_TOKEN or not PUSHOVER_USER:
        logging.warning("Pushover credentials not set. Skipping notification.")
        return False

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
                "priority": 0,
                "sound": "magic",
                "html": 1
            },
            timeout=10
        )

        response.raise_for_status()
        return True
    except requests.exceptions.RequestException as e:
        logging.error(f"Pushover notification failed: {str(e)}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error sending Pushover notification: {str(e)}")
        return False

async def send_pushover_notification_async(user_id: str, question: str, email: Optional[str] = None):
    """Asynchronous wrapper for Pushover notification"""
    await asyncio.to_thread(send_pushover_notification, user_id, question, email)

# ------------------ Agent Response Generator ------------------
async def process_agent_response(agent: str, question: str) -> Dict:
    """LLM decides if response should be a brief greeting or detailed help"""
    if gemini_model is None:
        logging.error(f"Agent {agent} error: Gemini model not initialized.")
        return {
            "agent": agent,
            "response": f"⚠️ {agent} is currently unavailable because the AI model could not be loaded. Please try again later.",
            "isTechnical": False
        }

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

        # Retry logic for Gemini API
        for attempt in range(3):
            try:
                response = gemini_model.generate_content(
                    prompt,
                    generation_config={
                        "temperature": 0.3,
                        "top_p": 0.95,
                        "max_output_tokens": 2048
                    }
                )
                break
            except Exception as e:
                if attempt < 2:
                    wait_time = 2 ** attempt
                    logging.warning(f"Retry {attempt + 1}/3 after {wait_time} sec due to: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    raise

        return {
            "agent": agent,
            "response": response.text,
            "isTechnical": True
        }

    except Exception as e:
        logging.error(f"Agent {agent} error during content generation: {e}")
        logging.debug(traceback.format_exc())
        return {
            "agent": agent,
            "response": f"⚠️ {agent} is currently experiencing issues. Please try again later.",
            "isTechnical": False
        }

# ------------------ Main Endpoint Logic ------------------
async def run_agents(req: AgentRequest):
    try:
        # Check critical dependencies first
        if db is None:
            logging.error("Run Agents: Firestore client is not initialized. Cannot proceed.")
            raise HTTPException(status_code=503, detail="Service Unavailable: Database connection failed during startup.")
        if gemini_model is None:
            logging.error("Run Agents: Gemini model is not initialized. Cannot proceed.")
            raise HTTPException(status_code=503, detail="Service Unavailable: AI model initialization failed during startup.")

        if len(req.agents) > 5:
            raise HTTPException(status_code=400, detail="Maximum 5 agents allowed")

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
        logging.debug(f"Session data written to {session_ref.id}")

        # Step 3: If reminders are enabled, save question and send notification
        if req.send_email:
            user_ref = db.collection("users").document(req.userId)
            user_ref.set({
                "reminderEnabled": True,
                "reminderQuestion": req.question.strip(),
                "lastUpdated": firestore.SERVER_TIMESTAMP,
                "email": req.email if req.email else None
            }, merge=True)

            logging.info(f"✅ reminderQuestion saved: {req.question}")

            # Send Pushover notification asynchronously
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

    except HTTPException:
        raise  # Re-raise FastAPI HTTPExceptions as-is
    except Exception as e:
        logging.error(f"Unexpected error during agent processing: {str(e)}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

# ------------------ Keyword Extractor ------------------
def extract_tech_keywords(text: str) -> List[str]:
    tech_terms = [
        'python', 'javascript', 'react', 'node', 'django', 'flask',
        'machine learning', 'ai', 'data science', 'database',
        'frontend', 'backend', 'fullstack', 'devops', 'kubernetes', 'docker',
        'aws', 'azure', 'google cloud', 'gcp', 'algorithms', 'data structures'
    ]
    return [term for term in tech_terms if term in text.lower()]

# ------------------ Health Check (for external use/monitoring) ------------------
async def health_check():
    """Performs health checks on critical services."""
    health = {
        "firestore": "unavailable",
        "gemini": "unavailable",
        "pushover": "unavailable"
    }

    # Firestore Check
    if db:
        try:
            health_ref = db.collection("health_checks").document("api_status")
            health_ref.set({"last_checked": firestore.SERVER_TIMESTAMP, "status": "ok_from_health_check"})
            health["firestore"] = "connected"
        except Exception as e:
            health["firestore"] = f"error: {str(e)}"
    
    # Gemini Check
    if gemini_model:
        try:
            test_gemini_response = gemini_model.generate_content("health check", generation_config={"max_output_tokens": 10})
            if test_gemini_response.text:
                health["gemini"] = "available"
        except Exception as e:
            health["gemini"] = f"error: {str(e)}"

    # Pushover Check
    pushover_ok = send_pushover_notification("health-check", "Health check notification")
    health["pushover"] = "connected" if pushover_ok else "unavailable"

    overall_status = "healthy" if all(status == "available" for status in health.values()) else "unhealthy"

    return {
        "status": overall_status,
        "services": health,
        "timestamp": datetime.now().isoformat()
    }
