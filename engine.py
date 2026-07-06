"""Ingestion, retrieval, and prompt-assembly logic for the Shufu Content Engine.

No LLM API calls happen anywhere in this module — embeddings are computed
locally with sentence-transformers, and the "prompt" produced here is just a
formatted string the user copies into claude.ai by hand.
"""

from __future__ import annotations

import csv
import json
import os
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np

REQUIRED_COLUMNS = [
    "source",
    "creator_handle",
    "platform",
    "theme",
    "hook",
    "script",
    "caption",
    "length_seconds",
    "shot_style",
    "views",
    "saves",
    "notes",
]

VALID_SHOT_STYLES = {"long_shot", "5_shots", "10plus_shots"}
VALID_SOURCES = {"own", "external"}

EMBEDDINGS_CACHE = Path("embeddings.npy")
INDEX_CACHE = Path("index.json")

_MODEL = None


class CSVValidationError(Exception):
    """Raised when posts.csv is missing required columns entirely."""


@dataclass
class Post:
    """A single content post (own or external) with its performance data."""

    row_number: int
    source: str
    creator_handle: str
    platform: str
    theme: str
    hook: str
    script: str
    caption: str
    length_seconds: int
    shot_style: str
    views: int
    saves: int
    notes: str

    def embedding_text(self) -> str:
        """Build the string used to compute this post's embedding."""
        return f"{self.theme} {self.hook} {self.caption} {self.script}".strip()

    def to_dict(self) -> dict:
        return asdict(self)


def generate_sample_csv(path: Path) -> None:
    """Write a 5-row sample posts.csv so the app is testable immediately."""
    rows = [
        {
            "source": "own",
            "creator_handle": "@shufu",
            "platform": "tiktok",
            "theme": "how to prepare shufu",
            "hook": "this is the only way I eat oatmeal now",
            "script": "[B-ROLL: pouring oats into jar] this is the only way I eat oatmeal now / [B-ROLL: adding hot water] no cooking, no dishes / [B-ROLL: stirring in toppings] just add whatever you've got / [B-ROLL: first bite] and it actually tastes good",
            "caption": "the 90 second breakfast that fixed my skin #shufu #guthealth #oatmeal",
            "length_seconds": 22,
            "shot_style": "5_shots",
            "views": 184000,
            "saves": 3200,
            "notes": "strong retention on first 3 seconds",
        },
        {
            "source": "own",
            "creator_handle": "@shufu",
            "platform": "instagram",
            "theme": "photo dump of casual shufu",
            "hook": "day in my life as a founder eating the same breakfast for 8 months",
            "script": "",
            "caption": "photo dump: what building Shufu actually looks like day to day",
            "length_seconds": 35,
            "shot_style": "10plus_shots",
            "views": 92000,
            "saves": 1400,
            "notes": "",
        },
        {
            "source": "external",
            "creator_handle": "@wellnesswithkay",
            "platform": "tiktok",
            "theme": "gut health morning routine",
            "hook": "if your skin is breaking out check your gut first",
            "script": "",
            "caption": "the gut-skin link nobody explains properly",
            "length_seconds": 41,
            "shot_style": "long_shot",
            "views": 512000,
            "saves": 8900,
            "notes": "single continuous talking head shot",
        },
        {
            "source": "own",
            "creator_handle": "@shufu",
            "platform": "tiktok",
            "theme": "founder story building in public",
            "hook": "I quit my job to sell oatmeal, here's month 1 revenue",
            "script": "[B-ROLL: laptop with spreadsheet] I quit my job to sell oatmeal / [B-ROLL: revenue screenshot] here's exactly what month 1 looked like / [B-ROLL: packing boxes] the good, the bad, and what's next",
            "caption": "month 1 numbers, no filter #buildinpublic #startup",
            "length_seconds": 28,
            "shot_style": "5_shots",
            "views": 231000,
            "saves": 5100,
            "notes": "",
        },
        {
            "source": "external",
            "creator_handle": "@thefoodlabnyc",
            "platform": "instagram",
            "theme": "5 minute recipes for busy people",
            "hook": "you have 5 minutes and zero motivation, make this",
            "script": "",
            "caption": "5 minute no-cook breakfast for people who hate mornings",
            "length_seconds": 18,
            "shot_style": "5_shots",
            "views": 76000,
            "saves": 980,
            "notes": "fast cuts, no talking head",
        },
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def load_posts(csv_path: Path) -> tuple[list[Post], list[str]]:
    """Load and validate posts.csv.

    Returns (posts, warnings). Raises CSVValidationError if required columns
    are missing entirely. Rows with invalid shot_style or unparseable numeric
    fields are skipped and reported in the returned warnings list.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
        if missing:
            raise CSVValidationError(
                f"posts.csv is missing required columns: {', '.join(missing)}"
            )

        posts: list[Post] = []
        warnings: list[str] = []

        for i, row in enumerate(reader, start=2):  # row 1 is the header
            reasons = []

            shot_style = (row.get("shot_style") or "").strip()
            if shot_style not in VALID_SHOT_STYLES:
                reasons.append(
                    f"invalid shot_style '{shot_style}' (must be one of {sorted(VALID_SHOT_STYLES)})"
                )

            source = (row.get("source") or "").strip()
            if source not in VALID_SOURCES:
                reasons.append(f"invalid source '{source}' (must be 'own' or 'external')")

            length_seconds_raw = (row.get("length_seconds") or "").strip()
            try:
                length_seconds = int(float(length_seconds_raw))
            except ValueError:
                reasons.append(f"invalid length_seconds '{length_seconds_raw}'")
                length_seconds = 0

            views_raw = (row.get("views") or "").strip()
            try:
                views = int(float(views_raw))
            except ValueError:
                reasons.append(f"invalid views '{views_raw}'")
                views = 0

            saves_raw = (row.get("saves") or "").strip()
            try:
                saves = int(float(saves_raw)) if saves_raw else 0
            except ValueError:
                reasons.append(f"invalid saves '{saves_raw}'")
                saves = 0

            theme = (row.get("theme") or "").strip()
            hook = (row.get("hook") or "").strip()
            if not theme or not hook:
                reasons.append("missing required theme or hook")

            if reasons:
                warnings.append(f"Row {i} skipped: {'; '.join(reasons)}")
                continue

            posts.append(
                Post(
                    row_number=i,
                    source=source,
                    creator_handle=(row.get("creator_handle") or "").strip(),
                    platform=(row.get("platform") or "").strip(),
                    theme=theme,
                    hook=hook,
                    script=(row.get("script") or "").strip(),
                    caption=(row.get("caption") or "").strip(),
                    length_seconds=length_seconds,
                    shot_style=shot_style,
                    views=views,
                    saves=saves,
                    notes=(row.get("notes") or "").strip(),
                )
            )

        if warnings:
            print("Warnings while loading posts.csv:")
            for w in warnings:
                print(f"  - {w}")

        return posts, warnings


def _get_model():
    """Lazily load the sentence-transformers model (slow import/load)."""
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _MODEL


def build_or_load_embeddings(
    posts: list[Post], csv_path: Path
) -> np.ndarray:
    """Compute embeddings for all posts, using an on-disk cache when fresh.

    The cache (embeddings.npy + index.json) is recomputed only when the
    CSV's modification time is newer than what's recorded in the cache, or
    when the row count no longer matches.
    """
    csv_mtime = csv_path.stat().st_mtime

    if EMBEDDINGS_CACHE.exists() and INDEX_CACHE.exists():
        try:
            with open(INDEX_CACHE) as f:
                meta = json.load(f)
            if meta.get("csv_mtime") == csv_mtime and meta.get("num_rows") == len(posts):
                return np.load(EMBEDDINGS_CACHE)
        except (json.JSONDecodeError, OSError):
            pass

    model = _get_model()
    texts = [p.embedding_text() for p in posts]
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)

    np.save(EMBEDDINGS_CACHE, embeddings)
    with open(INDEX_CACHE, "w") as f:
        json.dump({"csv_mtime": csv_mtime, "num_rows": len(posts)}, f)

    return embeddings


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string, normalized for cosine similarity via dot product."""
    model = _get_model()
    vec = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(vec, dtype=np.float32)[0]


def search(
    query: str, posts: list[Post], embeddings: np.ndarray, top_k: int = 10
) -> list[tuple[Post, float]]:
    """Return the top_k posts most similar to the query, by cosine similarity."""
    if not posts:
        return []
    query_vec = embed_query(query)
    scores = embeddings @ query_vec
    top_k = min(top_k, len(posts))
    top_indices = np.argsort(-scores)[:top_k]
    return [(posts[i], float(scores[i])) for i in top_indices]


def compute_stats(results: list[tuple[Post, float]]) -> dict:
    """Compute summary stats across a set of (post, score) results."""
    if not results:
        return {}

    posts = [p for p, _ in results]
    lengths = [p.length_seconds for p in posts]
    views = [p.views for p in posts]
    hook_word_counts = [len(p.hook.split()) for p in posts]

    shot_style_counts: dict[str, int] = {}
    for p in posts:
        shot_style_counts[p.shot_style] = shot_style_counts.get(p.shot_style, 0) + 1
    dominant_shot_style = max(shot_style_counts, key=shot_style_counts.get)

    own_count = sum(1 for p in posts if p.source == "own")
    external_count = sum(1 for p in posts if p.source == "external")

    return {
        "median_length_seconds": statistics.median(lengths),
        "min_length_seconds": min(lengths),
        "max_length_seconds": max(lengths),
        "shot_style_counts": shot_style_counts,
        "dominant_shot_style": dominant_shot_style,
        "median_views": statistics.median(views),
        "total_views": sum(views),
        "own_count": own_count,
        "external_count": external_count,
        "median_hook_word_count": statistics.median(hook_word_counts),
    }


def format_example(post: Post, score: Optional[float] = None) -> str:
    """Format a single post as a compact block for inclusion in the prompt."""
    lines = []
    prefix = f"[{score * 100:.0f}% match] " if score is not None else ""
    lines.append(f"{prefix}Source: {post.source} | Platform: {post.platform} | Creator: {post.creator_handle}")
    lines.append(f"Theme: {post.theme}")
    lines.append(f"Hook: {post.hook}")
    lines.append(f"Caption: {post.caption}")
    lines.append(
        f"Length: {post.length_seconds}s | Shot style: {post.shot_style} | Views: {post.views} | Saves: {post.saves}"
    )
    if post.script:
        lines.append(f"Script: {post.script}")
    if post.notes:
        lines.append(f"Notes: {post.notes}")
    return "\n".join(lines)


def format_examples_block(results: list[tuple[Post, float]]) -> str:
    """Format all top-10 examples as a single copy-pasteable block."""
    blocks = [format_example(p, score) for p, score in results]
    return "\n\n".join(f"Example {i+1}:\n{block}" for i, block in enumerate(blocks))


def assemble_prompt(
    query: str,
    results: list[tuple[Post, float]],
    stats: dict,
    brand_voice_text: str,
) -> str:
    """Assemble the full self-contained prompt for pasting into claude.ai."""
    examples_block = format_examples_block(results)

    shot_style_summary = ", ".join(
        f"{style}: {count}" for style, count in sorted(stats["shot_style_counts"].items())
    )

    stats_block = (
        f"Across these examples: median length {stats['median_length_seconds']:.0f}s "
        f"(range {stats['min_length_seconds']}-{stats['max_length_seconds']}s), "
        f"dominant shot style {stats['dominant_shot_style']} ({shot_style_summary}), "
        f"median views {stats['median_views']:.0f}, total views {stats['total_views']}, "
        f"own vs external: {stats['own_count']} vs {stats['external_count']}, "
        f"median hook length {stats['median_hook_word_count']:.0f} words."
    )

    task_block = (
        f"My idea: {query}. Using the examples above as style and structure reference, write: "
        "(1) a full script with [B-ROLL: description] markers at each cut, "
        "(2) a ready-to-post caption with hashtags, "
        "(3) recommended video length in seconds, "
        "(4) shot style (long_shot / 5_shots / 10plus_shots) with a shot list, "
        "(5) 2-3 sentences on why this should work, referencing the examples. "
        "Match the median length and dominant shot style unless my idea clearly demands otherwise."
    )

    return "\n\n".join(
        [
            brand_voice_text.strip(),
            "Here are 10 successful posts similar to my idea, with performance data:",
            examples_block,
            stats_block,
            task_block,
        ]
    )
