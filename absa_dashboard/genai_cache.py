"""
Simple file-based cache for GenAI Q&A results.

Stores (asin, normalized_question) -> answer, snippets in a small sqlite DB with TTL.

API:
 - init_cache(db_path)
 - get_cached_answer(asin, question)
 - set_cached_answer(asin, question, answer, snippets, ttl_seconds)
 - clear_cache()
"""
import sqlite3
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

DEFAULT_DB = str(Path(__file__).parent / 'genai_cache.db')


def init_cache(db_path: str = DEFAULT_DB):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_cache (
            id INTEGER PRIMARY KEY,
            asin TEXT NOT NULL,
            question TEXT NOT NULL,
            answer TEXT,
            snippets_json TEXT,
            created_at INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def normalize_question(q: str) -> str:
    return (q or '').strip().lower()


def get_cached_answer(asin: str, question: str, db_path: str = DEFAULT_DB, ttl_seconds: int = 7 * 24 * 3600) -> Optional[Dict[str, Any]]:
    init_cache(db_path)
    qnorm = normalize_question(question)
    now = int(time.time())
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "SELECT answer, snippets_json, created_at FROM qa_cache WHERE asin=? AND question=? ORDER BY created_at DESC LIMIT 1",
        (asin, qnorm),
    )
    row = cur.fetchone()
    conn.close()
    if not row:
        # record miss
        try:
            record_cache_event('miss', db_path=db_path)
        except Exception:
            pass
        return None
    answer, snippets_json, created_at = row
    if now - int(created_at) > ttl_seconds:
        # record miss (stale)
        try:
            record_cache_event('miss', db_path=db_path)
        except Exception:
            pass
        return None
    snippets = json.loads(snippets_json) if snippets_json else []
    # record hit
    try:
        record_cache_event('hit', db_path=db_path)
    except Exception:
        pass
    return {'answer': answer, 'snippets': snippets, 'created_at': int(created_at)}


def set_cached_answer(asin: str, question: str, answer: str, snippets: list, db_path: str = DEFAULT_DB):
    init_cache(db_path)
    qnorm = normalize_question(question)
    now = int(time.time())
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO qa_cache (asin, question, answer, snippets_json, created_at) VALUES (?, ?, ?, ?, ?)",
        (asin, qnorm, answer, json.dumps(snippets, ensure_ascii=False), now),
    )
    conn.commit()
    conn.close()


def clear_cache(db_path: str = DEFAULT_DB):
    if Path(db_path).exists():
        Path(db_path).unlink()
    init_cache(db_path)


def record_cache_event(event_type: str, db_path: str = DEFAULT_DB):
    """Record a cache event: 'hit' or 'miss' with timestamp."""
    if event_type not in ('hit', 'miss'):
        return
    init_cache(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "CREATE TABLE IF NOT EXISTS cache_events (id INTEGER PRIMARY KEY, event_type TEXT, ts INTEGER)"
    )
    now = int(time.time())
    cur.execute("INSERT INTO cache_events (event_type, ts) VALUES (?, ?)", (event_type, now))
    conn.commit()
    conn.close()


def cache_metrics(db_path: str = DEFAULT_DB) -> dict:
    """Return cache metrics: hit_count, miss_count, last_event_ts"""
    init_cache(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    # ensure events table exists
    cur.execute("CREATE TABLE IF NOT EXISTS cache_events (id INTEGER PRIMARY KEY, event_type TEXT, ts INTEGER)")
    cur.execute("SELECT COUNT(1) FROM cache_events WHERE event_type='hit'")
    hit = cur.fetchone()[0] or 0
    cur.execute("SELECT COUNT(1) FROM cache_events WHERE event_type='miss'")
    miss = cur.fetchone()[0] or 0
    cur.execute("SELECT MAX(ts) FROM cache_events")
    last = cur.fetchone()[0]
    conn.close()
    return {'hit_count': int(hit), 'miss_count': int(miss), 'last_event_ts': int(last) if last is not None else None}


def cache_stats(db_path: str = DEFAULT_DB) -> dict:
    """Return basic cache statistics: count and last_updated (unix ts)"""
    init_cache(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(1), MAX(created_at) FROM qa_cache")
    row = cur.fetchone()
    conn.close()
    if not row:
        return {'count': 0, 'last_updated': None}
    count, last = row
    return {'count': int(count or 0), 'last_updated': int(last) if last is not None else None}
