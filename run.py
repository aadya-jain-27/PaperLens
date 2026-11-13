"""Start PaperLens:  .venv/bin/python run.py   then open http://localhost:8000"""
import uvicorn

if __name__ == "__main__":
    print("\n  PaperLens running at  http://localhost:8000\n")
    uvicorn.run("paperlens.server:app", host="127.0.0.1", port=8000, log_level="warning")
