#!/usr/bin/env python3
"""Academic English Daily — save content to 2nd_brain and push to Feishu.

Usage:
    python3 run.py --input content.json   # read pre-generated content
    python3 run.py --dry-run              # skip feishu, only save locally
"""

import json
import os
import sys
import time
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

import requests

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).resolve().parent
ENV_FILE = PROJECT_DIR / ".env"
BRAIN_DIR = Path.home() / ".openclaw" / "workspace" / "2nd_brain" / "Language" / "academic-english"

# ── Feishu API ─────────────────────────────────────────────────────────────

def load_env():
    """Load .env into os.environ."""
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ[key.strip()] = val.strip().strip('"').strip("'")


def get_tenant_token(app_id: str, app_secret: str) -> str:
    """Get Feishu tenant_access_token."""
    resp = requests.post(
        "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal",
        json={"app_id": app_id, "app_secret": app_secret},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != 0:
        raise RuntimeError(f"Feishu auth failed: {data}")
    return data["tenant_access_token"]


def send_feishu_text(token: str, receive_id: str, text: str) -> dict:
    """Send a text message to a Feishu user via DM."""
    body = {
        "receive_id": receive_id,
        "msg_type": "text",
        "content": json.dumps({"text": text}),
    }
    resp = requests.post(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
        json=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=15,
    )
    return resp.json()


def send_feishu_post(token: str, receive_id: str, title: str, lines: list[str]) -> dict:
    """Send a rich-text (post) message to Feishu."""
    # Build post content: each line as a paragraph
    content_blocks = []
    for line in lines:
        # Replace markdown bold with feishu bold (if needed, but feishu post
        # doesn't support markdown natively — we just send as plain text blocks)
        content_blocks.append([{"tag": "text", "text": line}])

    post_content = {
        "zh_cn": {
            "title": title,
            "content": content_blocks,
        }
    }
    body = {
        "receive_id": receive_id,
        "msg_type": "post",
        "content": json.dumps(post_content, ensure_ascii=False),
    }
    resp = requests.post(
        "https://open.feishu.cn/open-apis/im/v1/messages?receive_id_type=open_id",
        json=body,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        timeout=15,
    )
    return resp.json()


# ── Content formatting ─────────────────────────────────────────────────────

def format_markdown(data: dict, date_str: str) -> str:
    """Convert JSON content data to a Markdown note."""
    day_num = data.get("day", "???")
    topic = data.get("topic", "AI/ML")
    
    md = f"# Day {day_num} — {date_str}\n\n"
    md += f"> 今日主题：**{topic}**\n\n---\n\n"

    # Vocabulary
    md += "## 📖 今日词汇\n\n"
    for i, word in enumerate(data.get("vocabulary", []), 1):
        w = word.get("word", "???")
        pron = word.get("pronunciation", "")
        meaning = word.get("meaning", "")
        example = word.get("example", "")
        note = word.get("note", "")
        md += f"### {i}. **{w}**" + (f" /{pron}/" if pron else "") + "\n\n"
        md += f"- **论文含义**：{meaning}\n"
        md += f"- **例句**：*{example}*\n"
        if note:
            md += f"- 💡 {note}\n"
        md += "\n"

    # Sentence patterns
    md += "## ✍️ 今日句式\n\n"
    for i, pat in enumerate(data.get("patterns", []), 1):
        p = pat.get("pattern", "???")
        usage = pat.get("usage", "")
        variants = pat.get("variants", [])
        md += f"### {i}. `{p}`\n\n"
        md += f"- **用途**：{usage}\n"
        if variants:
            md += "- **变体**：\n"
            for v in variants:
                md += f"  - `{v}`\n"
        md += "\n"

    # Writing tips
    md += "## 💡 写作技巧\n\n"
    for i, tip in enumerate(data.get("tips", []), 1):
        t = tip.get("tip", "???")
        detail = tip.get("detail", "")
        example_tip = tip.get("example", "")
        md += f"### {i}. {t}\n\n"
        if detail:
            md += f"{detail}\n\n"
        if example_tip:
            md += f"> 示例：{example_tip}\n\n"

    md += "---\n\n"
    md += f"*自动生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')} · Academic English Daily*\n"
    return md


def format_feishu_text(data: dict, date_str: str) -> str:
    """Format content as a compact Feishu text message."""
    day_num = data.get("day", "???")
    topic = data.get("topic", "AI/ML")
    
    lines = []
    lines.append(f"📚 学术英语日报 · Day {day_num} · {date_str}")
    lines.append(f"🎯 主题：{topic}")
    lines.append("")

    # Vocabulary (compact)
    lines.append("━━━ 📖 今日词汇 ━━━")
    for word in data.get("vocabulary", []):
        w = word.get("word", "?")
        pron = word.get("pronunciation", "")
        m = word.get("meaning", "?")
        line = f"• {w}" + (f" /{pron}/" if pron else "") + f" — {m}"
        lines.append(line)
    
    lines.append("")
    lines.append("━━━ ✍️ 今日句式 ━━━")
    for pat in data.get("patterns", [])[:5]:
        lines.append(f"• {pat.get('pattern', '?')}")
        lines.append(f"  {pat.get('usage', '')}")
    
    lines.append("")
    lines.append("━━━ 💡 写作技巧 ━━━")
    for tip in data.get("tips", [])[:3]:
        lines.append(f"• {tip.get('tip', '?')}")
        if tip.get("detail"):
            lines.append(f"  {tip['detail']}")
    
    lines.append("")
    lines.append("📥 完整笔记已存入 2nd_brain")
    
    return "\n".join(lines)


def update_history(content: dict):
    """Append today's words/patterns/tips to history.json (keep last 120 entries)."""
    hist_file = PROJECT_DIR / "history.json"
    hist = {"used_vocabulary": [], "used_patterns": [], "used_tips": []}
    if hist_file.exists():
        try:
            hist = json.loads(hist_file.read_text())
        except Exception:
            pass
    
    # Extract words (just the word strings)
    new_words = [w["word"] for w in content.get("vocabulary", [])]
    new_patterns = [p["pattern"] for p in content.get("patterns", [])]
    new_tips = [t["tip"] for t in content.get("tips", [])]
    
    # Append and deduplicate, keep last 120
    for key, new_items in [
        ("used_vocabulary", new_words),
        ("used_patterns", new_patterns),
        ("used_tips", new_tips),
    ]:
        existing = hist.get(key, [])
        combined = existing + new_items
        # Remove duplicates while preserving order
        seen = set()
        deduped = []
        for item in combined:
            if item not in seen:
                seen.add(item)
                deduped.append(item)
        hist[key] = deduped[-120:]
    
    hist["_comment"] = "最近约15天内出现过的词/句式/技巧会被自动排除"
    hist_file.write_text(json.dumps(hist, ensure_ascii=False, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Academic English Daily")
    parser.add_argument("--input", "-i", help="JSON content file")
    parser.add_argument("--dry-run", action="store_true", help="Skip Feishu send")
    parser.add_argument("--no-feishu", action="store_true", help="Skip Feishu send")
    args = parser.parse_args()

    load_env()

    # ── Load content ──
    if args.input:
        content = json.loads(Path(args.input).read_text())
    else:
        print("Error: --input required", file=sys.stderr)
        sys.exit(1)

    # ── Compute date ──
    tz = timezone(timedelta(hours=8))  # Asia/Shanghai
    now = datetime.now(tz)
    date_str = now.strftime("%B %d, %Y")
    date_path = now.strftime("%Y/%m-%d")

    # ── Save to 2nd_brain ──
    brain_path = BRAIN_DIR / date_path
    brain_path.mkdir(parents=True, exist_ok=True)
    md_content = format_markdown(content, date_str)
    md_file = brain_path / "daily.md"
    md_file.write_text(md_content)
    print(f"✅ Saved to {md_file}")

    # Update history for next day's dedup
    update_history(content)

    # Sync to Windows
    win_brain = Path("/mnt/c/Users/Damon/Documents/2nd_brain/Language/academic-english") / date_path
    win_brain.mkdir(parents=True, exist_ok=True)
    (win_brain / "daily.md").write_text(md_content)
    print(f"✅ Synced to Windows")

    # ── Send to Feishu ──
    if args.dry_run or args.no_feishu:
        print("⏭️  Skipped Feishu (dry-run / --no-feishu)")
    else:
        app_id = os.environ.get("FEISHU_APP_ID", "")
        app_secret = os.environ.get("FEISHU_APP_SECRET", "")
        receive_id = os.environ.get("FEISHU_RECEIVE_ID", "")

        if not all([app_id, app_secret, receive_id]):
            print("⚠️  Missing Feishu credentials, skipping send", file=sys.stderr)
        else:
            try:
                token = get_tenant_token(app_id, app_secret)
                feishu_text = format_feishu_text(content, date_str)
                result = send_feishu_text(token, receive_id, feishu_text)
                if result.get("code") == 0:
                    print(f"✅ Sent to Feishu: {result.get('data', {}).get('message_id', '?')}")
                else:
                    print(f"❌ Feishu send failed: {result}", file=sys.stderr)
            except Exception as e:
                print(f"❌ Feishu error: {e}", file=sys.stderr)

    print("🎉 Done!")


if __name__ == "__main__":
    main()
