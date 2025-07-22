from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import logging
import json

# Import functions and initialized clients from agentic_ai_backend.py
from agentic_ai_backend import (
    run_agents,
    send_pushover_notification,
    health_check as agent_health_check,  # Alias to avoid naming conflict with FastAPI's @app.get("/health")
    db,  # Firebase client from agentic_ai_backend
    gemini_model,  # Gemini model from agentic_ai_backend
    PUSHOVER_TOKEN,
    PUSHOVER_USER
)

# ------------------ Load Environment Variables ------------------
load_dotenv()

# ------------------ Firebase Initialization Check ------------------
if db is None:
    raise Exception("Firebase Firestore not initialized. Please check Firebase configuration.")

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
    # Add any other origins you need for development or testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers, including Content-Type, Authorization
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
        raise HTTPException(status_code=500, detail=f"An unexpected server error occurred: {str(e)}")

# ------------------ Async Notification Helper ------------------
async def send_pushover_notification_async(user_id: str, question: str, email: Optional[str] = None):
    """Wrapper to run Pushover notification in background"""
    try:
        await asyncio.sleep(1)  # Small delay to ensure main request completes first
        success = send_pushover_notification(user_id, question, email)
        if not success:
            logging.error("Pushover notification failed after async attempt")
    except Exception as e:
        logging.error(f"Async notification error: {str(e)}")
        logging.debug(traceback.format_exc())

# ------------------ Logging Setup ------------------
logging.basicConfig(level=logging.DEBUG)

# ------------------ App Entry Point (for local development) ------------------
if __name__ == "__main__":
    import uvicorn
    # Render uses the command in your "Start Command" field (e.g., uvicorn main:app --host 0.0.0.0 --port $PORT)
    # This block is primarily for local testing.
    uvicorn.run(app, host="0.0.0.0", port=8000)
