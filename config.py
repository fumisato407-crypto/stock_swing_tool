from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

WATCHLIST_PATH = BASE_DIR / "watchlist.csv"
TRADES_PATH = BASE_DIR / "trades.csv"

def get_setting(name: str, default: str = "") -> str:
    """Read local env vars first, then Streamlit Cloud secrets if available."""
    value = os.getenv(name)
    if value is not None:
        return value

    try:
        import streamlit as st

        secret_value = st.secrets.get(name, default)
        return str(secret_value) if secret_value is not None else default
    except Exception:
        return default


DEFAULT_PRICE_PERIOD = get_setting("PRICE_PERIOD", "9mo")
NOTIFICATION_CHANNEL = get_setting("NOTIFICATION_CHANNEL", "console")
OPENAI_API_KEY = get_setting("OPENAI_API_KEY", "")
OPENAI_MODEL = get_setting("OPENAI_MODEL", "gpt-4o-mini")
