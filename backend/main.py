from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pathlib import Path

from scanner import scan

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_FILE = BASE_DIR.parent / "frontend" / "index.html"


@app.get("/api/scan")
def run_scan():
    results = scan()

    return {
        "count": len(results),
        "results": results
    }


@app.get("/", response_class=HTMLResponse)
def home():
    return FRONTEND_FILE.read_text(
        encoding="utf-8"
    )
