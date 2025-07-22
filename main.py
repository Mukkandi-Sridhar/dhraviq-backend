from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import logging
import json
import asyncio
import traceback
from firebase_admin import firestore

# Import functions and initialized clients from agentic_ai_backend.py
from agentic_ai_backend import (
    run_agents,
    send_pushover_notification,
    health_check as agent_health_check,
    process_agent_response,
    send_pushover_notification_async,
    extract_tech_keywords,
    db,  # Firebase client
    gemini_model,
    PUSHOVER_TOKEN,
    PUSHOVER_USER,
    AgentRequest  # Assuming AgentRequest is defined here
)

# ------------------ Load Environment Variables ------------------
load_dotenv()

# ------------------ Firebase Initialization ------------------
# Load from Render secret or local file
secret_path = "/run/secrets/firebase-service-account.json"
if os.path.exists(secret_path):
    with open(secret_path, 'r') as f:
        cred_dict = json.load(f)
else:
    local_path = "firebase-service-account.json"
    with open(local_path, 'r') as f:
        cred_dict = json.load(f)

import firebase_admin
from firebase_admin import credentials
cred = credentials.Certificate(cred_dict)
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)
db = firestore.client()
logging.info("Firebase initialized successfully.")

# ------------------ Gemini API Key Check ------------------
if not os.getenv("GEMINI_API_KEY"):
    raise Exception("Gemini API key is missing. Please check environment variables.")

# ------------------ Pushover Token/User Check ------------------
if not PUSHOVER_TOKEN or not PUSHOVER_USER:
    raise Exception("Pushover credentials are missing. Please check environment variables.")

# ------------------ FastAPI App ------------------
app = FastAPI(
    title="Dhraviq Agentic AI Backend",
    description="Multi-agent orchestration using Gemini, Firestore & Pushover",
    version="1.0.0"
)

# ------------------ CORS Middleware ------------------
ALLOWED_ORIGINS = [
    "https://dhraviq.com",
    "https://www.dhraviq.com",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Health Check ------------------
@app.get("/health", include_in_schema=False)
@app.head("/health")
async def health_endpoint():
    logging.info("Health check started.")
    result = await agent_health_check()
    logging.info(f"Health check result: {result}")
    return result

# ------------------ Pushover Notification Test ------------------
@app.get("/test_notification", tags=["Diagnostics"])
def test_notification_endpoint():
    """Test endpoint for sending a Pushover notification."""
    success = send_pushover_notification(
        user_id="test-user",
        question="This is a test notification from /test_notification",
        email="test@example.com"
    )
    return {
        "sent": success,
        "message": (
            "✅ If 'sent' is true, you should receive a test notification "
            "on your Pushover mobile app."
        )
    }

# ------------------ Run Agents Endpoint ------------------
@app.post("/run_agents", tags=["Core Agents"])
async def run_agents_endpoint(req: AgentRequest):
    try:
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
        logging.error(f"Error occurred during agent processing: {str(e)}")
        logging.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

# ------------------ Logging Setup ------------------
logging.basicConfig(level=logging.DEBUG)

# ------------------ App Entry Point (for local development) ------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
