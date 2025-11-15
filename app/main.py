from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from .model import lifespan
from .api import router as api_router
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Phi-2 ChatML API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

@app.get("/", response_class=HTMLResponse)
async def get_homepage():
    html_file_path = Path(__file__).parent.parent / "frontend" / "index.html"
    
    if not html_file_path.exists():
        return HTMLResponse(
            content="<html><body><h1>Error 404</h1><p>File 'frontend/index.html' not found.</p></body></html>",
            status_code=404,
        )
    return HTMLResponse(content=html_file_path.read_text(), status_code=200)

STATIC_FILES_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=STATIC_FILES_DIR), name="static_frontend")