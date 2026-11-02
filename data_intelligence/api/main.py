"""
FastAPI Application Entry Point.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from data_intelligence.config import settings
from data_intelligence.api.routes import router

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Autonomous Multi-Model Data Intelligence Agent Platform for SMBs",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for web clients and dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "message": f"Welcome to {settings.app_name}",
        "version": settings.app_version,
        "docs": "/docs",
        "api_v1": "/api/v1",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("data_intelligence.api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
