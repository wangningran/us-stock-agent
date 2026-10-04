"""数据层：日线行情 + 分析师评级/目标价变动（免费源 yfinance），带本地 CSV 缓存。"""
from __future__ import annotations

import time
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


def load_universe(cfg: dict) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], pd.DataFrame]:
    """返回 (prices, events, benchmark)。单只股票失败时跳过并提示。"""
    start, cache = cfg["data"]["start"], cfg["data"]["cache_dir"]
    prices, events = {}, {}
    for t in cfg["universe"]:
        try:
            prices[t] = get_prices(t, start, cache)
            events[t] = get_analyst_events(t, cache)
        except Exception as e:  # noqa: BLE001 - 单只失败不影响整体
            print(f"[warn] skip {t}: {e}")
    bench = get_prices(cfg["benchmark"], start, cache)
    return prices, events, bench
