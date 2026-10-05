"""Short-term mean reversion (high win-rate style) and short moving-average trend rules, with a signal backtester.

Reference rules:
- Larry Connors RSI(2): close > 200-day SMA and RSI(2) < 10 -> buy; close > 5-day SMA -> sell
- Cumulative RSI(2): two-day RSI(2) sum < 35 -> buy; RSI(2) > 65 -> sell
- IBS (close position within the day's range) < 0.2 on a down day -> buy; close above prior high -> sell
- EMA6 / EMA12 crossover (short moving-average trend following)

Execution modes:
- close: fill at the signal day's close (market-on-close order placed ~15 minutes before the close)
- open:  fill at the next day's open (orders placed from a pre-market report)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .signals import market_ok


def wilder_rsi(c: pd.Series, n: int) -> pd.Series:
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(100)


def mr_features(px: pd.DataFrame) -> pd.DataFrame:
    df = px.copy()
    c = df["close"]
    df["sma5"] = c.rolling(5).mean()
    df["sma200"] = c.rolling(200).mean()
    df["ema6"] = c.ewm(span=6, adjust=False).mean()
    df["ema12"] = c.ewm(span=12, adjust=False).mean()
    df["rsi2"] = wilder_rsi(c, 2)
    df["cum_rsi2"] = df["rsi2"] + df["rsi2"].shift(1)
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["ibs"] = ((c - df["low"]) / rng).fillna(0.5)
    df["mom63"] = c / c.shift(63) - 1
    return df


# Each rule: entry(df) / exit(df) return bool Series evaluated at the close; lower rank(df) is preferred
RULES = {
    "rsi2": dict(
        entry=lambda d: (d.close > d.sma200) & (d.rsi2 < 10),
        exit=lambda d: d.close > d.sma5,
        rank=lambda d: d.rsi2),
    "rsi2_strict": dict(
        entry=lambda d: (d.close > d.sma200) & (d.rsi2 < 5),
        exit=lambda d: d.close > d.sma5,
        rank=lambda d: d.rsi2),
    "cum_rsi2": dict(
        entry=lambda d: (d.close > d.sma200) & (d.cum_rsi2 < 35),
        exit=lambda d: d.rsi2 > 65,
        rank=lambda d: d.cum_rsi2),
    "ibs": dict(
        entry=lambda d: (d.close > d.sma200) & (d.ibs < 0.2) & (d.close < d.close.shift(1)),
        exit=lambda d: d.close > d.high.shift(1),
        rank=lambda d: d.ibs),
    "ema6_12_cross": dict(
        entry=lambda d: (d.close > d.sma200) & (d.ema6 > d.ema12) & (d.ema6.shift(1) <= d.ema12.shift(1)),
        exit=lambda d: d.ema6 < d.ema12,
        rank=lambda d: -d.mom63),
    "ema_trend_rsi2_dip": dict(   # sharp dip inside a short-term uptrend
        entry=lambda d: (d.close > d.sma200) & (d.ema6 > d.ema12) & (d.rsi2 < 15),
        exit=lambda d: d.close > d.sma5,
        rank=lambda d: d.rsi2),
}


def build_mr(prices: dict[str, pd.DataFrame], rule: str, members: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    r = RULES[rule]
    out = {}
    for t, px in prices.items():
        d = mr_features(px)
        member = members[t].reindex(d.index).fillna(False).astype(bool) if members is not None and t in members \
            else pd.Series(members is None, index=d.index)
        d["entry"] = (r["entry"](d) & member & d.sma200.notna()).fillna(False)
        d["exit"] = r["exit"](d).fillna(False)
        d["rank"] = r["rank"](d)
        out[t] = d
    return out


def run_mr_backtest(feats: dict[str, pd.DataFrame], bench: pd.DataFrame, *, mode: str = "close",
                    max_positions: int = 10, time_stop: int = 10, stop_pct: float | None = None,
                    slippage_bps: float = 5, commission: float = 1.0, equity0: float = 100_000,
                    market_filter: bool = False, start=None, end=None, max_new_per_day: int | None = None):
    """Signal backtest with equal weights: each position = equity / max_positions."""
    days = bench.index
    if start:
        days = days[days >= pd.Timestamp(start)]
    if end:
        days = days[days < pd.Timestamp(end)]
    mkt = market_ok(bench)
    slip = slippage_bps / 1e4
    tk = list(feats)
    P = {k: pd.DataFrame({t: feats[t][k] for t in tk}).reindex(days)
         for k in ("open", "high", "low", "close", "rank")}
    ENTRY = pd.DataFrame({t: feats[t]["entry"] for t in tk}).reindex(days).fillna(False).astype(bool)
    EXIT = pd.DataFrame({t: feats[t]["exit"] for t in tk}).reindex(days).fillna(False).astype(bool)
    closes_ff = P["close"].ffill()

    cash, pos, trades, curve = equity0, {}, [], {}
    pend_buy, pend_sell = [], set()

    def sell(t, px, day, why):
        nonlocal cash
        p = pos.pop(t)
        fill = px * (1 - slip)
        cash += p["sh"] * fill - commission
        trades.append(dict(ticker=t, entry_date=p["d"], entry=p["px"], exit_date=day, exit=fill,
                           ret=fill / p["px"] - 1, pnl=(fill - p["px"]) * p["sh"] - 2 * commission,
                           days=p["n"], reason=why))

    def buy(t, px, day, equity):
        nonlocal cash
        fill = px * (1 + slip)
        sh = int(min(equity / max_positions, cash - commission) // fill)
        if sh < 1:
            return
        cash -= sh * fill + commission
        pos[t] = dict(px=fill, sh=sh, d=day, n=0)

    for day in days:
        o, h, l, c = P["open"].loc[day], P["high"].loc[day], P["low"].loc[day], P["close"].loc[day]
        # Next-open execution (open mode)
        if mode == "open":
            for t in list(pend_sell):
                if t in pos and pd.notna(o[t]):
                    sell(t, o[t], day, "signal")
            pend_sell.clear()
            eq_open = cash + sum(p["sh"] * closes_ff.at[day, t] for t, p in pos.items())
            for t in pend_buy:
                if t not in pos and len(pos) < max_positions and pd.notna(o[t]):
                    buy(t, o[t], day, eq_open)
            pend_buy = []
        # Intraday stop
        if stop_pct:
            for t in list(pos):
                p = pos[t]
                stop = p["px"] * (1 - stop_pct)
                if pd.notna(l[t]) and l[t] <= stop and p["d"] != day:
                    sell(t, min(o[t], stop), day, "stop")
        # At the close: update holding days, check exits
        for t in list(pos):
            p = pos[t]
            if p["d"] == day or pd.isna(c[t]):
                continue
            p["n"] += 1
            if EXIT.at[day, t] or p["n"] >= time_stop:
                why = "signal" if EXIT.at[day, t] else "time"
                if mode == "close":
                    sell(t, c[t], day, why)
                else:
                    pend_sell.add(t)
        equity = cash + sum(p["sh"] * closes_ff.at[day, t] for t, p in pos.items())
        curve[day] = equity
        # At the close: new signals
        if market_filter and not bool(mkt.get(day, False)):
            continue
        slots = max_positions - len(pos) + (len(pend_sell) if mode == "open" else 0)
        if slots <= 0:
            continue
        cands = ENTRY.columns[ENTRY.loc[day].values]
        cands = [t for t in cands if t not in pos]
        cands.sort(key=lambda t: (pd.isna(P["rank"].at[day, t]), P["rank"].at[day, t]))
        if max_new_per_day:
            slots = min(slots, max_new_per_day)
        for t in cands[:slots]:
            if mode == "close":
                buy(t, c[t], day, equity)
            else:
                pend_buy.append(t)

    last = closes_ff.iloc[-1]
    for t in list(pos):
        sell(t, last[t], days[-1], "end")
    tr = pd.DataFrame(trades)
    return tr, pd.Series(curve, name="equity")


def mr_summary(tr: pd.DataFrame, eq: pd.Series) -> dict:
    if tr.empty:
        return {"trades": 0}
    w, lz = tr[tr.pnl > 0], tr[tr.pnl <= 0]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    return dict(
        trades=len(tr), win_rate=len(w) / len(tr),
        avg_win=w.ret.mean() if len(w) else 0, avg_loss=lz.ret.mean() if len(lz) else 0,
        avg_ret=tr.ret.mean(), pf=w.pnl.sum() / abs(lz.pnl.sum()) if len(lz) else np.inf,
        days=tr.days.mean(), cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
        mdd=(eq / eq.cummax() - 1).min(), sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan,
        worst=tr.ret.min(),
    )


# ---------------- Extended indicators used as filters on top of RSI(2) ----------------

def ext_features(px: pd.DataFrame) -> pd.DataFrame:
    """mr_features plus MACD, Bollinger Bands, ADX, volume, gap and volatility features."""
    d = mr_features(px)
    c, h, l = d["close"], d["high"], d["low"]
    d["sma20"] = c.rolling(20).mean()
    d["sma50"] = c.rolling(50).mean()
    sd20 = c.rolling(20).std()
    d["bb_lower"] = d["sma20"] - 2 * sd20
    d["bb_pctb"] = (c - d["bb_lower"]) / (4 * sd20)          # Bollinger %B; < 0 means below the lower band
    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    d["atr"] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    d["atr_pct"] = d["atr"] / c
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=d.index)
    ndm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=d.index)
    atr_w = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    pdi = 100 * pdm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    ndi = 100 * ndm.ewm(alpha=1 / 14, adjust=False).mean() / atr_w
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    d["adx"] = dx.ewm(alpha=1 / 14, adjust=False).mean()
    d["vol_ratio"] = d["volume"] / d["volume"].rolling(20).mean()
    d["gap"] = d["open"] / prev_c - 1
    d["ret1"] = c / prev_c - 1
    d["dd10"] = c / h.rolling(10).max() - 1                     # drawdown from the 10-day high
    low14, high14 = l.rolling(14).min(), h.rolling(14).max()
    d["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    d["down_days"] = (c < prev_c).astype(int).groupby((c >= prev_c).cumsum()).cumsum()
    return d


def market_context(bench: pd.DataFrame, vix: pd.DataFrame | None) -> pd.DataFrame:
    c = bench["close"]
    m = pd.DataFrame(index=bench.index)
    m["spy_above200"] = c > c.rolling(200).mean()
    m["spy_above50"] = c > c.rolling(50).mean()
    m["spy_rsi2"] = wilder_rsi(c, 2)
    m["vix"] = vix["close"].reindex(bench.index).ffill() if vix is not None else np.nan
    return m


def build_custom(base: dict[str, pd.DataFrame], mkt: pd.DataFrame, entry, exit_, rank,
                 members: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """base: dict of ext_features frames; entry / exit_ / rank take (d, mkt) and return Series."""
    out = {}
    for t, d0 in base.items():
        d = d0.copy()
        m = mkt.reindex(d.index)
        member = members[t].reindex(d.index).fillna(False).astype(bool) if members is not None and t in members \
            else pd.Series(members is None, index=d.index)
        d["entry"] = (entry(d, m) & member & d.sma200.notna()).fillna(False).astype(bool)
        d["exit"] = exit_(d, m).fillna(False).astype(bool)
        d["rank"] = rank(d, m)
        out[t] = d
    return out


def add_dollar_volume_rank(base: dict[str, pd.DataFrame], index: pd.DatetimeIndex,
                           members: pd.DataFrame | None = None, window: int = 60) -> None:
    """Rank index members each day by `window`-day average dollar volume; writes base[t]["dv_rank"] (1 = largest).

    A point-in-time proxy for size / popularity that only uses data known at the time (no look-ahead).
    """
    dv = pd.DataFrame({t: (d.close * d.volume).rolling(window).mean() for t, d in base.items()}).reindex(index)
    if members is not None:
        dv = dv.where(members.reindex(index=index, columns=dv.columns).fillna(False).astype(bool))
    rank = dv.rank(axis=1, ascending=False)
    for t in base:
        base[t]["dv_rank"] = rank[t].reindex(base[t].index)
