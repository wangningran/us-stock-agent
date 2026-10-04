"""历史成分股（point-in-time）：每个交易日只允许交易当天确实在标普 500 里的股票。

数据来源：github.com/fja05680/sp500（基于 Andreas Clenow《Trading Evolved》的成分股历史，持续更新）。
局限：Yahoo 不提供已退市/已更名股票的行情，这部分股票在回测中缺失，覆盖率见 coverage()。
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
    """返回 [date, tickers(list)]，每行是该日起生效的成分股快照。"""
    path = Path(path) if path else ROOT / "data" / "ref" / "sp500_hist.csv"
    if not path.exists():
        import urllib.request
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SP500_URL, path)
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df["tickers"] = df["tickers"].map(lambda s: sorted({to_yahoo(t) for t in s.split(",") if t.strip()}))
    return df.reset_index(drop=True)


def tickers_between(hist: pd.DataFrame, start: str, end: str | None = None) -> list[str]:
    """[start, end] 期间曾经是成分股的全部代码（含区间开始前最后一次快照）。"""
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
    """bool 矩阵 [days × tickers]：当天是否为成分股。快照向前填充到下一次变动。"""
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
