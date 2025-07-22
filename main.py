from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv

# Import functions and initialized clients from agentic_ai_backend.py
# Firebase initialization and 'db' client will now be handled within agentic_ai_backend.py
from agentic_ai_backend import (
    run_agents,
    send_pushover_notification,
    health_check as agent_health_check # Alias to avoid naming conflict with FastAPI's @app.get("/health")
)

# ------------------ Load Environment Variables ------------------
# This should be at the very top to ensure all environment variables are loaded
# before they are accessed in any imported modules.
load_dotenv()

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
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers, including Content-Type, Authorization
)

# ------------------ Health Check ------------------
# This health check now calls the one defined in agentic_ai_backend.py
# which includes checks for Firestore and Pushover
@app.get("/health", include_in_schema=False)
@app.head("/health")
async def health_endpoint():
    # Call the health check function from agentic_ai_backend
    return await agent_health_check()


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
# The actual logic for run_agents is now fully contained in agentic_ai_backend.py
# We are simply exposing it as a FastAPI POST endpoint here.
app.post("/run_agents", tags=["Core Agents"])(run_agents)

# ------------------ App Entry Point (for local development) ------------------
if __name__ == "__main__":
    import uvicorn
    # Render uses the command in your "Start Command" field (e.g., uvicorn main:app --host 0.0.0.0 --port $PORT)
    # This block is primarily for local testing.
    uvicorn.run(app, host="0.0.0.0", port=8000)
