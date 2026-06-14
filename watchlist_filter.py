from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from data_fetcher import normalize_jp_symbol


def normalize_symbol(symbol: str) -> str:
    text = str(symbol or "").strip().upper()
    if not text:
        return ""
    return normalize_jp_symbol(text) or ""


def parse_symbol_input(text: str) -> List[str]:
    if not str(text or "").strip():
        return []
    parts = re.split(r"[\s,\u3000]+", str(text))
    symbols: List[str] = []
    seen = set()
    for part in parts:
        normalized = normalize_symbol(part)
        if normalized and normalized not in seen:
            symbols.append(normalized)
            seen.add(normalized)
    return symbols


def _watchlist_symbol(row: pd.Series) -> str:
    return normalize_symbol(row.get("normalized_symbol") or row.get("code") or row.get("raw_code") or "")


def _symbol_label(row: pd.Series, symbol: str) -> str:
    name = str(row.get("name", "") or "").strip()
    return f"{symbol} {name}".strip()


def filter_watchlist_symbols(
    watchlist: pd.DataFrame,
    include_symbols: Optional[Iterable[str]] = None,
    exclude_symbols: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    source = watchlist.copy() if isinstance(watchlist, pd.DataFrame) else pd.DataFrame()
    original_count = int(len(source))
    if source.empty:
        return {
            "filtered_watchlist": source,
            "included_symbols": [],
            "excluded_symbols": [],
            "excluded_labels": [],
            "missing_symbols": list(include_symbols or []) + list(exclude_symbols or []),
            "original_count": original_count,
            "included_count": 0,
            "final_count": 0,
        }

    source["_filter_symbol"] = source.apply(_watchlist_symbol, axis=1)
    symbol_to_index = {
        str(row["_filter_symbol"]): idx
        for idx, row in source.iterrows()
        if str(row.get("_filter_symbol", ""))
    }
    watchlist_symbols = set(symbol_to_index.keys())

    include_list = [normalize_symbol(symbol) for symbol in (include_symbols or [])]
    include_list = [symbol for symbol in include_list if symbol]
    exclude_list = [normalize_symbol(symbol) for symbol in (exclude_symbols or [])]
    exclude_list = [symbol for symbol in exclude_list if symbol]

    if include_list:
        include_set = set(include_list)
        working = source[source["_filter_symbol"].isin(include_set)].copy()
        missing_include = [symbol for symbol in include_list if symbol not in watchlist_symbols]
        included_symbols = [symbol for symbol in include_list if symbol in watchlist_symbols]
    else:
        working = source.copy()
        missing_include = []
        included_symbols = working["_filter_symbol"].dropna().astype(str).tolist()

    included_count = int(len(working))
    exclude_set = set(exclude_list)
    excluded_rows = working[working["_filter_symbol"].isin(exclude_set)].copy()
    excluded_symbols = excluded_rows["_filter_symbol"].dropna().astype(str).tolist()
    excluded_labels = [_symbol_label(row, str(row["_filter_symbol"])) for _, row in excluded_rows.iterrows()]
    filtered = working[~working["_filter_symbol"].isin(exclude_set)].copy()
    missing_exclude = [symbol for symbol in exclude_list if symbol not in watchlist_symbols]
    missing_symbols = list(dict.fromkeys(missing_include + missing_exclude))

    filtered = filtered.drop(columns=["_filter_symbol"], errors="ignore")
    return {
        "filtered_watchlist": filtered,
        "included_symbols": list(dict.fromkeys(included_symbols)),
        "excluded_symbols": list(dict.fromkeys(excluded_symbols)),
        "excluded_labels": excluded_labels,
        "missing_symbols": missing_symbols,
        "original_count": original_count,
        "included_count": included_count,
        "final_count": int(len(filtered)),
    }
