from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from scanner import scan

app = FastAPI()


@app.get("/api/scan")
def run_scan():

    results = scan()

    return {
        "count": len(results),
        "results": results
    }


@app.get("/", response_class=HTMLResponse)
def home():

    with open("index.html", "r", encoding="utf-8") as file:
        return file.read()
