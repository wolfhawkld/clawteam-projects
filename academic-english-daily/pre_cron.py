#!/usr/bin/env python3
"""Pre-cron script: outputs day counter, topic, and avoidance context for the cron agent."""

import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

PROJECT_DIR = Path.home() / ".openclaw" / "workspace" / "clawteam-projects" / "academic-english-daily"
COUNTER_FILE = PROJECT_DIR / "day_counter.txt"
TOPICS_FILE = PROJECT_DIR / "topics.json"
HISTORY_FILE = PROJECT_DIR / "history.json"

tz = timezone(timedelta(hours=8))
today = datetime.now(tz).strftime("%Y-%m-%d")


def load_topics():
    """Load topic list from JSON, fallback to hardcoded list."""
    if TOPICS_FILE.exists():
        try:
            data = json.loads(TOPICS_FILE.read_text())
            topics = data.get("topics", [])
            if topics:
                return topics
        except Exception:
            pass
    return ["AI/ML Research Papers — Academic English"]


def get_day_number() -> tuple[str, int]:
    """Read or init day counter. Returns (today_str, day_number)."""
    if COUNTER_FILE.exists():
        lines = COUNTER_FILE.read_text().strip().splitlines()
        if len(lines) >= 2 and lines[0] == today:
            return today, int(lines[1])
        else:
            return today, (int(lines[1]) + 1 if len(lines) >= 2 else 1)
    return today, 1


def save_day_number(day_num: int):
    COUNTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    COUNTER_FILE.write_text(f"{today}\n{day_num}")


def load_history(max_recent: int = 100):
    """Load recently used vocabulary words / patterns / tips."""
    if HISTORY_FILE.exists():
        try:
            data = json.loads(HISTORY_FILE.read_text())
            return {
                "vocab": data.get("used_vocabulary", [])[-max_recent:],
                "patterns": data.get("used_patterns", [])[-max_recent:],
                "tips": data.get("used_tips", [])[-max_recent:],
            }
        except Exception:
            pass
    return {"vocab": [], "patterns": [], "tips": []}


# ── Main ──

today_str, day_num = get_day_number()
save_day_number(day_num)

topics = load_topics()
topic = topics[(day_num - 1) % len(topics)]

history = load_history()

# Build output for cron agent
output = {
    "day": str(day_num).zfill(3),
    "topic": topic,
    "date": today_str,
    "project_dir": str(PROJECT_DIR),
    "avoid_vocabulary": history["vocab"],
    "avoid_patterns": history["patterns"],
    "avoid_tips": history["tips"],
}

print(json.dumps(output, ensure_ascii=False))
