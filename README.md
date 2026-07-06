# ML Content Engine

A local retrieval and prompt-building tool for short-form video content. Finds the 10 most similar posts from `posts.csv`, and shows their
performance stats. Runs locally.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

Open http://localhost:8000

## Data

Put your posts in `posts.csv` in the project root (see columns in the code /
`engine.py`). If `posts.csv` doesn't exist, a 5-row sample is generated
automatically on first run so the app is testable immediately.

Embeddings are cached to `embeddings.npy` and `index.json` and only
recomputed when `posts.csv` changes. Use the "Reload data" button in the UI
to re-ingest the CSV without restarting the server.

Edit `brand_voice.md` to change the brand rules baked into every generated
prompt — no code changes needed.
