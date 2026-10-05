"""Point-in-time S&P 500 membership: each day only stocks that were index members that day are tradable.

Source: github.com/fja05680/sp500 (membership history from Andreas Clenow's "Trading Evolved", kept up to date).
Limitation: Yahoo has no prices for delisted / renamed tickers, so those are missing from backtests.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import ROOT

SP500_URL = ("https://raw.githubusercontent.com/fja05680/sp500/master/"
             "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv")


def to_yahoo(ticker: str) -> str:
    return ticker.strip().replace(".", "-")


def load_sp500_history(path: str | Path | None = None) -> pd.DataFrame:
    """Return [date, tickers(list)]; each row is the membership snapshot effective from that date."""
    path = Path(path) if path else ROOT / "data" / "ref" / "sp500_hist.csv"
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SP500_URL, path)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df["tickers"] = df["tickers"].map(lambda s: sorted({to_yahoo(t) for t in s.split(",") if t.strip()}))
    return df.reset_index(drop=True)


def tickers_between(hist: pd.DataFrame, start: str, end: str | None = None) -> list[str]:
    """All tickers that were members during [start, end] (including the last snapshot before start)."""
    start = pd.Timestamp(start)
    first = hist.index[hist.date <= start]
    lo = first[-1] if len(first) else 0
    sub = hist.iloc[lo:]
    if end:
        sub = sub[sub.date <= pd.Timestamp(end)]
    out: set[str] = set()
    for ts in sub.tickers:
        out.update(ts)
    return sorted(out)


def membership(hist: pd.DataFrame, days: pd.DatetimeIndex, tickers: list[str]) -> pd.DataFrame:
    """Boolean [days x tickers] matrix of membership; each snapshot is forward-filled until the next change."""
    snap_idx = hist.date.searchsorted(days, side="right") - 1
    cols = {t: i for i, t in enumerate(tickers)}
    import numpy as np
    mat = np.zeros((len(days), len(tickers)), dtype=bool)
    snaps = [set(ts) for ts in hist.tickers]
    for row, si in enumerate(snap_idx):
        if si < 0:
            continue
        for t in snaps[si]:
            j = cols.get(t)
            if j is not None:
                mat[row, j] = True
    return pd.DataFrame(mat, index=days, columns=tickers)


def current_members(hist: pd.DataFrame) -> list[str]:
    return list(hist.tickers.iloc[-1])
