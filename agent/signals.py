"""Signal layer (legacy v1 analyst strategy): technical indicators, analyst event scoring and candidate filters.

All signals are computed after day t's close using data up to day t only; trades execute on day t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

BULLISH_GRADES = {
    "buy", "strong buy", "outperform", "overweight", "positive", "accumulate",
    "sector outperform", "market outperform", "add", "conviction buy", "top pick",
    "long-term buy", "outperformer",
}
BEARISH_GRADES = {
    "sell", "strong sell", "underperform", "underweight", "negative", "reduce",
    "sector underperform", "market underperform",
}


def indicators(px: pd.DataFrame, mom_lookback: int = 63) -> pd.DataFrame:
    df = px.copy()
    c = df["close"]
    df["sma20"] = c.rolling(20).mean()
    df["sma50"] = c.rolling(50).mean()
    df["sma200"] = c.rolling(200).mean()
    prev_c = c.shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_c).abs(),
                    (df["low"] - prev_c).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    delta = c.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(100)
    df["mom"] = c / c.shift(mom_lookback) - 1
    return df


def _pt_change(row) -> float | None:
    cur, prior = row.get("pt_current"), row.get("pt_prior")
    try:
        cur, prior = float(cur), float(prior)
    except (TypeError, ValueError):
        return None
    if not (cur > 0 and prior > 0):
        return None
    return cur / prior - 1


def event_score_v2(row, pt_min_change: float = 0.10) -> int:
    """v2: count only meaningful actions.

    - Upgrade +2 / downgrade -2; bullish initiation +1 / bearish initiation -1
    - Price-target moves count (+/-1) only when |change| >= pt_min_change; "raises" of unknown size score 0
    - Plain reiterations and small target tweaks are treated as noise (0)
    """
    s = 0
    action = str(row.get("action") or "").lower()
    to_grade = str(row.get("to_grade") or "").strip().lower()
    if action == "up":
        s += 2
    elif action == "down":
        s -= 2
    elif action == "init":
        s += 1 if to_grade in BULLISH_GRADES else -1 if to_grade in BEARISH_GRADES else 0
    chg = _pt_change(row)
    if chg is not None:
        if chg >= pt_min_change:
            s += 1
        elif chg <= -pt_min_change:
            s -= 1
    return s


def event_score(row: pd.Series) -> int:
    """v1: score of one analyst event (any price-target raise counts)."""
    s = 0
    action = str(row.get("action") or "").lower()
    to_grade = str(row.get("to_grade") or "").strip().lower()
    if action == "up":
        s += 2
    elif action == "down":
        s -= 2
    elif action == "init":
        s += 1 if to_grade in BULLISH_GRADES else -1 if to_grade in BEARISH_GRADES else 0

    pt_action = str(row.get("pt_action") or "").lower()
    cur, prior = row.get("pt_current"), row.get("pt_prior")
    if pt_action == "raises":
        s += 1
    elif pt_action == "lowers":
        s -= 1
    elif pd.notna(cur) and pd.notna(prior) and prior and prior > 0:
        s += 1 if cur > prior else -1 if cur < prior else 0
    return s


def score_events(events: pd.DataFrame, scoring: str = "v1", pt_min_change: float = 0.10) -> pd.Series:
    if events is None or events.empty:
        return pd.Series(dtype=float)
    if scoring == "v2":
        return events.apply(lambda r: event_score_v2(r, pt_min_change), axis=1)
    return events.apply(event_score, axis=1)


def daily_analyst_score(events: pd.DataFrame, trading_days: pd.DatetimeIndex,
                        scoring: str = "v1", pt_min_change: float = 0.10) -> pd.Series:
    """Map events to trading days (treated as known after that day's close) and sum per day.

    Events published on weekends / holidays are assigned to the next trading day.
    """
    out = pd.Series(0.0, index=trading_days)
    if events is None or events.empty:
        return out
    scores = score_events(events, scoring, pt_min_change)
    dates = pd.to_datetime(events.index).normalize()
    pos = trading_days.searchsorted(dates, side="left")
    valid = pos < len(trading_days)
    agg = pd.Series(scores.values[valid], index=trading_days[pos[valid]]).groupby(level=0).sum()
    out.loc[agg.index] = agg
    return out


def market_ok(bench: pd.DataFrame) -> pd.Series:
    """Market filter: benchmark close above its 200-day SMA."""
    c = bench["close"]
    return c > c.rolling(200).mean()


def build_features(prices: dict[str, pd.DataFrame], events: dict[str, pd.DataFrame],
                   cfg: dict, members: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """members: [date x ticker] membership matrix; when given, non-members cannot be candidates that day."""
    sc = cfg["signals"]
    scoring, pt_min = sc.get("scoring", "v1"), sc.get("pt_min_change", 0.10)
    feats = {}
    for t, px in prices.items():
        df = indicators(px, sc["mom_lookback"])
        daily = daily_analyst_score(events.get(t), df.index, scoring, pt_min)
        df["analyst_score"] = daily.rolling(sc["analyst_lookback_days"], min_periods=1).sum()
        df["analyst_today"] = daily
        if members is not None and t in members.columns:
            df["member"] = members[t].reindex(df.index).fillna(False).astype(bool)
        else:
            df["member"] = members is None
        df["candidate"] = (
            df["member"] &
            (df["analyst_score"] >= sc["min_analyst_score"])
            & (df["close"] > df["sma50"])
            & (df["sma50"] > df["sma200"])
            & (df["mom"] > 0)
            & (df["rsi"] < sc["rsi_max"])
            & df["atr"].notna()
        )
        feats[t] = df
    return feats


def rank_candidates(feats: dict[str, pd.DataFrame], day: pd.Timestamp) -> list[tuple[str, pd.Series]]:
    """Candidates on a given day, sorted by analyst score then momentum (descending)."""
    rows = []
    for t, df in feats.items():
        if day in df.index and bool(df.at[day, "candidate"]):
            rows.append((t, df.loc[day]))
    rows.sort(key=lambda x: (x[1]["analyst_score"], x[1]["mom"]), reverse=True)
    return rows
