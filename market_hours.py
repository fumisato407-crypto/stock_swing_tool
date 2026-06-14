from __future__ import annotations

from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo


JST = ZoneInfo("Asia/Tokyo")
MORNING_START = time(9, 0)
MORNING_END = time(11, 30)
AFTERNOON_START = time(12, 30)
AFTERNOON_END = time(15, 30)


def now_jst() -> datetime:
    return datetime.now(JST)


def _as_jst(value: datetime | None = None) -> datetime:
    current = value or now_jst()
    if current.tzinfo is None:
        return current.replace(tzinfo=JST)
    return current.astimezone(JST)


def is_weekday_jst(value: datetime | None = None) -> bool:
    return _as_jst(value).weekday() < 5


def is_morning_session(value: datetime | None = None) -> bool:
    current = _as_jst(value)
    return is_weekday_jst(current) and MORNING_START <= current.time() < MORNING_END


def is_afternoon_session(value: datetime | None = None) -> bool:
    current = _as_jst(value)
    return is_weekday_jst(current) and AFTERNOON_START <= current.time() < AFTERNOON_END


def is_lunch_break(value: datetime | None = None) -> bool:
    current = _as_jst(value)
    return is_weekday_jst(current) and MORNING_END <= current.time() < AFTERNOON_START


def is_market_open_jst(value: datetime | None = None) -> bool:
    current = _as_jst(value)
    return is_morning_session(current) or is_afternoon_session(current)


def market_status_label(value: datetime | None = None) -> str:
    current = _as_jst(value)
    if not is_weekday_jst(current):
        return "土日"
    if is_morning_session(current):
        return "市場時間中（前場）"
    if is_lunch_break(current):
        return "昼休み"
    if is_afternoon_session(current):
        return "市場時間中（後場）"
    return "時間外"


def next_market_open_hint(value: datetime | None = None) -> str:
    current = _as_jst(value)
    today_morning = current.replace(hour=9, minute=0, second=0, microsecond=0)
    today_afternoon = current.replace(hour=12, minute=30, second=0, microsecond=0)

    if not is_weekday_jst(current):
        days_ahead = 1
        next_day = current + timedelta(days=days_ahead)
        while next_day.weekday() >= 5:
            days_ahead += 1
            next_day = current + timedelta(days=days_ahead)
        next_open = next_day.replace(hour=9, minute=0, second=0, microsecond=0)
    elif current < today_morning:
        next_open = today_morning
    elif current.time() < AFTERNOON_START:
        next_open = today_afternoon
    else:
        next_day = current + timedelta(days=1)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        next_open = next_day.replace(hour=9, minute=0, second=0, microsecond=0)

    return next_open.strftime("%Y-%m-%d %H:%M JST")
