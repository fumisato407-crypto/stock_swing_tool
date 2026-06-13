from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd


JST = ZoneInfo("Asia/Tokyo")


def now_jst() -> datetime:
    return datetime.now(JST)


def now_jst_iso() -> str:
    return now_jst().isoformat(timespec="seconds")


def now_jst_display() -> str:
    return now_jst().strftime("%Y-%m-%d %H:%M:%S JST")


def parse_trade_datetime_to_jst_naive(value: Any) -> Optional[datetime]:
    if value in (None, ""):
        return None
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    try:
        if ts.tzinfo is None:
            return ts.to_pydatetime().replace(tzinfo=None)
        return ts.tz_convert(JST).tz_localize(None).to_pydatetime()
    except Exception:
        try:
            py_dt = datetime.fromisoformat(str(value))
        except (TypeError, ValueError):
            return None
        if py_dt.tzinfo is None:
            return py_dt
        return py_dt.astimezone(JST).replace(tzinfo=None)
