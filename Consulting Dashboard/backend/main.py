"""
FastAPI entry point.
Run with:  uvicorn backend.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import CORS_ORIGINS, BACKEND_HOST, BACKEND_PORT
from backend.routers import papers, status, chat

app = FastAPI(
    title="Quant Research Dashboard API",
    description="Backend for the Quant Research Dashboard — bridges to Research_LLM",
    version="0.1.0",
)

# ── CORS ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────
app.include_router(papers.router)
app.include_router(status.router)
app.include_router(chat.router)


# ── Health check ───────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "quant-dashboard-api"}


# ── Dev entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host=BACKEND_HOST, port=BACKEND_PORT, reload=True)
