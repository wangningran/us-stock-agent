"""合成数据：离线跑通流程 / 单元测试用，不代表任何真实行情。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_prices(n_days: int = 900, seed: int = 0, drift: float = 0.0004, vol: float = 0.018,
                start: str = "2022-01-03", base: float = 100.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n_days)
    rets = rng.normal(drift, vol, n_days)
    close = base * np.exp(np.cumsum(rets))
    open_ = close * np.exp(rng.normal(0, vol / 3, n_days))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, vol / 2, n_days)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, vol / 2, n_days)))
    vol_ = rng.integers(1_000_000, 5_000_000, n_days)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol_}, index=idx)


def make_events(px: pd.DataFrame, n: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1000)
    days = rng.choice(px.index[200:], size=n, replace=False)
    actions = rng.choice(["up", "down", "main", "init", "reit"], size=n, p=[.3, .15, .35, .1, .1])
    rows = []
    for d, a in zip(sorted(days), actions):
        p = float(px.at[d, "close"])
        raise_ = a in ("up", "init") or (a == "main" and rng.random() < .6)
        rows.append({
            "time": pd.Timestamp(d) + pd.Timedelta(hours=12), "firm": f"Broker{rng.integers(1, 9)}",
            "to_grade": "Buy" if a != "down" else "Hold", "from_grade": "Hold" if a == "up" else "",
            "action": a, "pt_action": "Raises" if raise_ else "Lowers",
            "pt_current": round(p * (1.2 if raise_ else 0.9), 0), "pt_prior": round(p * 1.05, 0),
        })
    return pd.DataFrame(rows).set_index("time")


def make_universe(n_tickers: int = 12, n_days: int = 900, seed: int = 0):
    prices, events = {}, {}
    for i in range(n_tickers):
        t = f"SYN{i:02d}"
        prices[t] = make_prices(n_days, seed + i, drift=0.0002 + 0.0001 * (i % 5))
        events[t] = make_events(prices[t], seed=seed + i)
    bench = make_prices(n_days, seed + 999, drift=0.0004, vol=0.011, base=400)
    return prices, events, bench
