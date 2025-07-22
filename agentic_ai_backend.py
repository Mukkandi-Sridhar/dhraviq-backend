from fastapi import HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import asyncio
import os
import json # New import for handling JSON string
from dotenv import load_dotenv
import firebase_admin
from firebase_admin import firestore, initialize_app, credentials
import google.generativeai as genai
import traceback
from datetime import datetime
import requests

# ------------------ Load Environment Variables ------------------
# Ensure this is loaded before any os.getenv calls are made
load_dotenv()

# ------------------ Firebase Initialization ------------------
# Get Firebase service account key from environment variable
FIREBASE_SERVICE_ACCOUNT_KEY_JSON = os.getenv('FIREBASE_SERVICE_ACCOUNT_KEY')

if FIREBASE_SERVICE_ACCOUNT_KEY_JSON:
    try:
        # Parse the JSON string from the environment variable
        cred_dict = json.loads(FIREBASE_SERVICE_ACCOUNT_KEY_JSON)
        cred = credentials.Certificate(cred_dict)
        # Initialize Firebase Admin SDK only once
        if not firebase_admin._apps:
            initialize_app(cred)
        db = firestore.client()
        print("Firebase Admin SDK initialized successfully from environment variable.")
    except json.JSONDecodeError as e:
        print(f"Error decoding FIREBASE_SERVICE_ACCOUNT_KEY JSON: {e}")
        print("Please ensure the FIREBASE_SERVICE_ACCOUNT_KEY environment variable on Render contains valid JSON.")
        raise Exception(f"Firebase JSON decode error: {e}") # Raise to halt deployment if invalid
    except Exception as e:
        print(f"Error during Firebase Admin SDK initialization: {e}")
        raise Exception(f"Firebase initialization failed: {e}") # Raise to halt deployment if invalid
else:
    print("WARNING: FIREBASE_SERVICE_ACCOUNT_KEY environment variable not found.")
    print("Firebase Admin SDK will NOT be initialized. Firestore operations will fail.")
    # In a production setup, you might want to raise an exception here
    # to prevent the application from running without critical credentials.
    # For now, we'll let it proceed but expect Firestore errors.
    db = None # Set db to None if not initialized
    raise Exception("FIREBASE_SERVICE_ACCOUNT_KEY environment variable is missing. Cannot initialize Firebase.")


# Gemini Model Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not found.")
    raise Exception("GEMINI_API_KEY environment variable is missing. Gemini will not work.")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-pro")

# Pushover Configuration
PUSHOVER_TOKEN = os.getenv("PUSHOVER_TOKEN")
PUSHOVER_USER = os.getenv("PUSHOVER_USER")
if not PUSHOVER_TOKEN or not PUSHOVER_USER:
    print("WARNING: Pushover API keys (PUSHOVER_TOKEN, PUSHOVER_USER) not found. Notifications will not work.")


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
        print("Pushover credentials not set. Skipping notification.")
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
                "priority": 0,  # Normal priority
                "sound": "magic",  # Custom notification sound
                "html": 1  # Enable basic HTML formatting
            },
            timeout=10  # 10 second timeout
        )

        response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
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
            "isTechnical": True # Assuming agent responses are generally technical
        }

    except Exception as e:
        print(f"Agent {agent} error: {e}")
        traceback.print_exc() # Print full traceback for agent errors
        return {
            "agent": agent,
            "response": f"⚠️ {agent} is currently unavailable due to an internal error. Please try again later.",
            "isTechnical": False
        }

# ------------------ Main Endpoint Logic ------------------
async def run_agents(req: AgentRequest):
    try:
        if db is None:
            raise HTTPException(500, detail="Firestore client not initialized. Check Firebase credentials.")

        if len(req.agents) > 5:
            raise HTTPException(400, detail="Maximum 5 agents allowed")

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
        session_ref.set(session_data) # This is line 181 from your original trace

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
            # Note: asyncio.create_task requires an async function
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
        raise # Re-raise FastAPI HTTPExceptions as-is
    except Exception as e:
        traceback.print_exc() # Print full traceback for unexpected errors
        raise HTTPException(500, detail=f"An unexpected server error occurred: {str(e)}")

# ------------------ Async Notification Helper ------------------
async def send_pushover_notification_async(user_id: str, question: str, email: Optional[str] = None):
    """Wrapper to run Pushover notification in background"""
    try:
        # Small delay to ensure main request completes first
        await asyncio.sleep(1)
        success = send_pushover_notification(user_id, question, email)
        if not success:
            print("Pushover notification failed after async attempt")
    except Exception as e:
        print(f"Async notification error: {str(e)}")
        traceback.print_exc()

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
    firestore_status = "disconnected"
    pushover_status = "unavailable"
    gemini_status = "unavailable"

    try:
        # Test Firestore connectivity by writing a temporary document
        if db is not None:
            health_ref = db.collection("health_checks").document("api_status")
            health_ref.set({"last_checked": firestore.SERVER_TIMESTAMP, "status": "ok"})
            firestore_status = "connected"
        else:
            firestore_status = "initialization_failed"

    except Exception as e:
        print(f"Firestore health check failed: {e}")
        firestore_status = f"error: {str(e)}"

    try:
        # Test Gemini connectivity (a simple query)
        # Note: A real health check for LLMs might be more robust
        test_gemini_response = gemini_model.generate_content("hello")
        if test_gemini_response.text:
            gemini_status = "available"
        else:
            gemini_status = "no_response"
    except Exception as e:
        print(f"Gemini health check failed: {e}")
        gemini_status = f"error: {str(e)}"

    # Test Pushover connectivity
    pushover_ok = send_pushover_notification(
        user_id="health-check",
        question="This is a test notification from the backend health check.",
        email="health@example.com"
    )
    pushover_status = "connected" if pushover_ok else "unavailable"


    # Return comprehensive status
    return {
        "status": "healthy" if firestore_status == "connected" and gemini_status == "available" and pushover_status != "unavailable" else "unhealthy",
        "services": {
            "firestore": firestore_status,
            "gemini": gemini_status,
            "pushover": pushover_status
        },
        "timestamp": datetime.now().isoformat()
    }
