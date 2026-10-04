"""数据层：日线行情 + 分析师评级/目标价变动（免费源 yfinance），带本地 CSV 缓存。"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .config import ROOT

PRICE_COLS = ["open", "high", "low", "close", "volume"]
EVENT_COLS = ["firm", "to_grade", "from_grade", "action",
              "pt_action", "pt_current", "pt_prior"]


def _cache_path(cache_dir: str, kind: str, ticker: str) -> Path:
    d = ROOT / cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{kind}_{ticker}.csv"


def _fresh(path: Path, max_age_hours: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < max_age_hours * 3600


def get_prices(ticker: str, start: str, cache_dir: str = "data",
               max_age_hours: float = 6) -> pd.DataFrame:
    """复权日线 OHLCV，索引为日期。"""
    path = _cache_path(cache_dir, "px", ticker)
    if _fresh(path, max_age_hours):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf
    df = yf.Ticker(ticker).history(start=start, auto_adjust=True)
    if df.empty:
        raise RuntimeError(f"no price data for {ticker}")
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df = df.rename(columns=str.lower)[PRICE_COLS]
    df.to_csv(path)
    return df


def get_analyst_events(ticker: str, cache_dir: str = "data",
                       max_age_hours: float = 6) -> pd.DataFrame:
    """分析师评级/目标价变动。索引为事件时间戳。

    action: up / down / init / main / reit
    pt_action: Raises / Lowers / Maintains / Announces（部分记录为空）
    """
    path = _cache_path(cache_dir, "analyst", ticker)
    if _fresh(path, max_age_hours):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf
    try:
        raw = yf.Ticker(ticker).upgrades_downgrades
    except Exception:
        raw = pd.DataFrame()
    df = normalize_events(raw)
    df.to_csv(path)
    return df


def normalize_events(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=EVENT_COLS, index=pd.DatetimeIndex([], name="time"))
    df = raw.rename(columns={
        "Firm": "firm", "ToGrade": "to_grade", "FromGrade": "from_grade",
        "Action": "action", "priceTargetAction": "pt_action",
        "currentPriceTarget": "pt_current", "priorPriceTarget": "pt_prior",
    })
    for c in EVENT_COLS:
        if c not in df.columns:
            df[c] = None
    df = df[EVENT_COLS].copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "time"
    return df.sort_index()


def _missing_registry(cache_dir: str) -> tuple[Path, dict]:
    path = ROOT / cache_dir / "missing.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    week_ago = time.time() - 7 * 86400
    return path, {t: ts for t, ts in data.items() if ts > week_ago}


def _fetch_one(t: str, start: str, cache: str):
    try:
        return t, get_prices(t, start, cache), get_analyst_events(t, cache)
    except Exception:  # noqa: BLE001 - 已退市/无数据的股票直接跳过
        return t, None, None


def load_universe(cfg: dict):
    """返回 (prices, events, benchmark, members)。

    universe_mode = "sp500_pit"：历史标普 500 成分股，members 为 [日期 × 代码] 的成分股矩阵；
    universe_mode = "list"：使用 config 中的 universe 列表，members 为 None。
    """
    start, cache = cfg["data"]["start"], cfg["data"]["cache_dir"]
    mode = cfg.get("universe_mode", "list")
    hist = None
    if mode == "sp500_pit":
        from .universe import load_sp500_history, tickers_between
        hist = load_sp500_history()
        tickers = tickers_between(hist, start)
    elif mode == "sp500_current":
        from .universe import current_members, load_sp500_history
        tickers = current_members(load_sp500_history())
    else:
        tickers = list(cfg["universe"])

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    reg_path, missing = _missing_registry(cache)
    todo = [t for t in tickers if t not in missing]
    prices, events = {}, {}
    with ThreadPoolExecutor(max_workers=cfg["data"].get("workers", 8)) as pool:
        for t, px, ev in pool.map(lambda t: _fetch_one(t, start, cache), todo):
            if px is None or px.empty:
                missing[t] = time.time()
            else:
                prices[t], events[t] = px, ev
    reg_path.write_text(json.dumps(missing))
    print(f"[data] {len(prices)}/{len(tickers)} 只股票有数据（缺失 {len(tickers) - len(prices)}，多为已退市/更名）")

    bench = get_prices(cfg["benchmark"], start, cache)
    members = None
    if hist is not None:
        from .universe import membership
        members = membership(hist, bench.index, sorted(prices))
    return prices, events, bench, members
