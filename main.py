"""FastAPI app for the Shufu Content Engine.

Serves a single-page UI for retrieving similar past posts and assembling a
copy-pasteable prompt. No LLM API calls happen here.
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import engine

CSV_PATH = Path("posts.csv")
BRAND_VOICE_PATH = Path("brand_voice.md")

app = FastAPI(title="Shufu Content Engine")
templates = Jinja2Templates(directory="templates")

state: dict = {
    "posts": [],
    "embeddings": None,
    "warnings": [],
    "error": None,
    "loaded": False,
}


def ingest() -> None:
    """Load posts.csv (generating a sample if missing) and refresh embeddings."""
    state["error"] = None
    state["warnings"] = []
    state["loaded"] = False

    if not CSV_PATH.exists():
        engine.generate_sample_csv(CSV_PATH)
        state["warnings"].append("posts.csv was missing, generated posts_sample.csv content as posts.csv.")

    try:
        posts, warnings = engine.load_posts(CSV_PATH)
    except engine.CSVValidationError as e:
        state["error"] = str(e)
        return

    state["warnings"].extend(warnings)

    if not posts:
        state["error"] = "posts.csv contains no valid rows after validation."
        return

    embeddings = engine.build_or_load_embeddings(posts, CSV_PATH)

    state["posts"] = posts
    state["embeddings"] = embeddings
    state["loaded"] = True


@app.on_event("startup")
def on_startup() -> None:
    ingest()


class SearchRequest(BaseModel):
    query: str


class SearchResult(BaseModel):
    similarity: float
    post: dict


class SearchResponse(BaseModel):
    results: list[SearchResult]
    stats: dict
    prompt: str
    examples_text: str


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.get("/api/status")
def status():
    return {
        "loaded": state["loaded"],
        "num_posts": len(state["posts"]),
        "warnings": state["warnings"],
        "error": state["error"],
    }


@app.post("/api/reload")
def reload_data():
    ingest()
    return status()


@app.post("/api/search", response_model=SearchResponse)
def search(req: SearchRequest):
    if not state["loaded"]:
        return SearchResponse(results=[], stats={}, prompt="", examples_text="")

    results = engine.search(req.query, state["posts"], state["embeddings"], top_k=10)
    stats = engine.compute_stats(results)
    brand_voice_text = BRAND_VOICE_PATH.read_text() if BRAND_VOICE_PATH.exists() else ""
    prompt = engine.assemble_prompt(req.query, results, stats, brand_voice_text)
    examples_text = engine.format_examples_block(results)

    return SearchResponse(
        results=[
            SearchResult(similarity=score, post=post.to_dict()) for post, score in results
        ],
        stats=stats,
        prompt=prompt,
        examples_text=examples_text,
    )
