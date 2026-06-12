from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

try:
    from dotenv import dotenv_values, load_dotenv

    load_dotenv(ENV_PATH, override=False)
except ImportError:
    dotenv_values = None
    pass

WATCHLIST_PATH = BASE_DIR / "watchlist.csv"
TRADES_PATH = BASE_DIR / "trades.csv"
ALERTS_LOG_PATH = BASE_DIR / "alerts_log.csv"

def get_setting(name: str, default: str = "") -> str:
    """Read settings in priority order: Streamlit Secrets, env vars, then .env."""
    try:
        import streamlit as st

        secret_value = st.secrets.get(name)
        if secret_value is not None and str(secret_value).strip():
            return str(secret_value)
    except Exception:
        pass

    value = os.getenv(name)
    if value is not None and str(value).strip():
        return str(value)

    try:
        if dotenv_values is not None and ENV_PATH.exists():
            env_file_value = dotenv_values(ENV_PATH).get(name)
            if env_file_value is not None and str(env_file_value).strip():
                return str(env_file_value)
    except Exception:
        pass

    return default


DEFAULT_PRICE_PERIOD = get_setting("PRICE_PERIOD", "6mo")
NOTIFICATION_CHANNEL = get_setting("NOTIFICATION_CHANNEL", "console")
DISCORD_WEBHOOK_URL = get_setting("DISCORD_WEBHOOK_URL", "")
DISCORD_MENTION_ID = get_setting("DISCORD_MENTION_ID", "")
OPENAI_API_KEY = get_setting("OPENAI_API_KEY", "")
OPENAI_MODEL = get_setting("OPENAI_MODEL", "gpt-4o-mini")
