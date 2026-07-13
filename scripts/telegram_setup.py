"""One-shot Telegram setup: verify a bot token, discover the chat id,
write both into .env, and send a test message.

Usage:
    uv run python scripts/telegram_setup.py <bot_token>

Before running, create the bot via @BotFather and send it any message
(bots can only reply, so the chat id comes from your first message).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
TIMEOUT = 15


def _api(token: str, method: str, **params: object) -> dict:
    response = requests.post(
        f"https://api.telegram.org/bot{token}/{method}",
        json=params,
        timeout=TIMEOUT,
    )
    data = response.json()
    if not data.get("ok"):
        raise SystemExit(f"Telegram API {method} failed: {data.get('description')}")
    return data["result"]


def _discover_chat_id(token: str) -> str:
    updates = _api(token, "getUpdates")
    chat_ids = {
        str(update["message"]["chat"]["id"])
        for update in updates
        if "message" in update
    }
    if not chat_ids:
        raise SystemExit(
            "No messages found. Open Telegram, send any message to your bot, "
            "then rerun this script."
        )
    if len(chat_ids) > 1:
        raise SystemExit(f"Multiple chats found ({chat_ids}); set TELEGRAM_CHAT_ID manually.")
    return chat_ids.pop()


def _write_env(token: str, chat_id: str) -> None:
    env_path = ROOT / ".env"
    text = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
    for key, value in (("TELEGRAM_BOT_TOKEN", token), ("TELEGRAM_CHAT_ID", chat_id)):
        line = f"{key}={value}"
        pattern = re.compile(rf"^#?\s*{key}=.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(line, text, count=1)
        else:
            text = text.rstrip("\n") + f"\n{line}\n"
    env_path.write_text(text, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    token = sys.argv[1].strip()
    bot = _api(token, "getMe")
    print(f"Token OK: bot @{bot['username']}")
    chat_id = _discover_chat_id(token)
    print(f"Chat id: {chat_id}")
    _write_env(token, chat_id)
    print("Wrote TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to .env")
    _api(
        token,
        "sendMessage",
        chat_id=chat_id,
        text="✅ Job agent connected. New job matches will arrive here.",
    )
    print("Test message sent — check your Telegram.")


if __name__ == "__main__":
    main()
