"""Volatility-squeeze / base breakout strategy (research, separate from the live RSI(2) strategy).

Two layers:
  watch    - setup before a breakout: Bollinger squeeze, above a rising 50-day SMA, contracting range with
             higher lows, drying volume, close within 3% of the base high, stronger than SPY.
  breakout - close above the prior 20-day high on above-average volume (optionally only after a recent setup).
Exits: intraday stop (breakout-day low, base midpoint or a fixed %), or a close below the 10/20-day SMA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def bo_features(px: pd.DataFrame, spy_close: pd.Series, *, base_n: int = 20, squeeze_q: float = 0.2,
                near_pct: float = 0.03) -> pd.DataFrame:
    d = px[["open", "high", "low", "close", "volume"]].copy()
    c, h, l, v = d.close, d.high, d.low, d.volume
    spy = spy_close.reindex(d.index).ffill()
    for n in (10, 20, 50, 200):
        d[f"sma{n}"] = c.rolling(n).mean()
    d["sma50_up"] = d.sma50 > d.sma50.shift(10)
    bbw = 4 * c.rolling(20).std() / d.sma20
    d["bbw"] = bbw
    d["bbw_pct"] = bbw.rolling(60, min_periods=60).rank(pct=True)
    d["squeeze"] = d.bbw_pct <= squeeze_q
    hi_incl, lo_incl = h.rolling(base_n).max(), l.rolling(base_n).min()
    d["base_hi"] = hi_incl.shift(1)                      # prior N-day high (breakout level)
    d["base_lo"] = lo_incl.shift(1)
    rng = (hi_incl - lo_incl) / c
    d["contracting"] = rng < rng.shift(base_n)           # last N days narrower than the N days before
    d["higher_lows"] = l.rolling(10).min() > l.shift(10).rolling(20).min()
    d["dry"] = v.rolling(5).mean() < v.rolling(20).mean()
    d["near_high"] = (c >= hi_incl * (1 - near_pct)) & (c <= d.base_hi)
    d["rs63"] = c / c.shift(63) - spy / spy.shift(63)
    d["rs21"] = c / c.shift(21) - spy / spy.shift(21)
    d["vol_ratio"] = v / v.rolling(20).mean().shift(1)
    trend = (c > d.sma50) & d.sma50_up
    d["watch"] = (d["squeeze"] & trend & d.contracting & d.higher_lows & d.dry & d.near_high & (d.rs63 > 0)).fillna(False)
    d["setup_recent"] = d.watch.shift(1).rolling(5, min_periods=1).max().fillna(0).astype(bool)
    d["squeeze_recent"] = d["squeeze"].shift(1).rolling(5, min_periods=1).max().fillna(0).astype(bool)
    d["breakout_raw"] = (c > d.base_hi) & trend & (d.rs63 > 0)
    return d


def run_bo_backtest(feats: dict[str, pd.DataFrame], days: pd.DatetimeIndex, *, entry_col: str = "entry",
                    stop: str = "day_low", max_stop: float = 0.10, exit_sma: int = 10, max_positions: int = 5,
                    max_new_per_day: int = 3, slippage_bps: float = 5, equity0: float = 5260,
                    mode: str = "close", max_hold: int | None = None):
    """Signal backtest. entry at the signal-day close (mode='close') or next open; equal-weight slots.

    stop: 'day_low' (breakout-day low), 'base_mid' (midpoint of the prior 20-day range), 'pct' (= max_stop).
    The stop is never farther than max_stop below the entry.
    """
    slip = slippage_bps / 1e4
    tk = list(feats)
    F = {k: pd.DataFrame({t: feats[t][k] for t in tk}).reindex(days)
         for k in ("open", "high", "low", "close", "base_hi", "base_lo", "rs63", f"sma{exit_sma}")}
    E = pd.DataFrame({t: feats[t][entry_col] for t in tk}).reindex(days).fillna(False).astype(bool)
    cff = F["close"].ffill()
    cash, pos, trades, curve, pend = equity0, {}, [], {}, []

    def sell(t, px, day, why):
        nonlocal cash
        p = pos.pop(t)
        fill = px * (1 - slip)
        cash += p["sh"] * fill
        trades.append(dict(ticker=t, entry_date=p["d"], entry=p["px"], exit_date=day, exit=fill,
                           ret=fill / p["px"] - 1, pnl=(fill - p["px"]) * p["sh"], days=p["n"], reason=why,
                           risk=1 - p["stop"] / p["px"]))

    def buy(t, px, day, equity, sig_day):
        nonlocal cash
        fill = px * (1 + slip)
        sh = int(min(equity / max_positions, cash) // fill)
        if sh < 1:
            return
        if stop == "day_low":
            s = F["low"].at[sig_day, t]
        elif stop == "base_mid":
            s = (F["base_hi"].at[sig_day, t] + F["base_lo"].at[sig_day, t]) / 2
        else:
            s = 0.0
        s = max(s if pd.notna(s) else 0.0, fill * (1 - max_stop))
        if s >= fill:
            s = fill * (1 - max_stop)
        cash -= sh * fill
        pos[t] = dict(px=fill, sh=sh, d=day, n=0, stop=s)

    for day in days:
        o, lo, c = F["open"].loc[day], F["low"].loc[day], F["close"].loc[day]
        if mode == "open":
            eq_o = cash + sum(p["sh"] * cff.at[day, t] for t, p in pos.items())
            for t, sd in pend:
                if t not in pos and len(pos) < max_positions and pd.notna(o[t]):
                    buy(t, o[t], day, eq_o, sd)
            pend = []
        for t in list(pos):                                   # intraday stop
            p = pos[t]
            if p["d"] != day and pd.notna(lo[t]) and lo[t] <= p["stop"]:
                sell(t, min(o[t], p["stop"]), day, "stop")
        for t in list(pos):                                   # close: trailing SMA exit
            p = pos[t]
            if p["d"] == day or pd.isna(c[t]):
                continue
            p["n"] += 1
            if c[t] < F[f"sma{exit_sma}"].at[day, t]:
                sell(t, c[t], day, "sma")
            elif max_hold and p["n"] >= max_hold:
                sell(t, c[t], day, "time")
        equity = cash + sum(p["sh"] * cff.at[day, t] for t, p in pos.items())
        curve[day] = equity
        slots = min(max_positions - len(pos), max_new_per_day)
        if slots <= 0:
            continue
        cands = [t for t in E.columns[E.loc[day].values] if t not in pos]
        cands.sort(key=lambda t: -(F["rs63"].at[day, t] if pd.notna(F["rs63"].at[day, t]) else -9))
        for t in cands[:slots]:
            if mode == "close":
                buy(t, c[t], day, equity, day)
            else:
                pend.append((t, day))
    for t in list(pos):
        sell(t, cff.iloc[-1][t], days[-1], "end")
    return pd.DataFrame(trades), pd.Series(curve, name="equity")


def bo_summary(tr: pd.DataFrame, eq: pd.Series) -> dict:
    if tr.empty:
        return {"trades": 0}
    w, lz = tr[tr.pnl > 0], tr[tr.pnl <= 0]
    yrs = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    r = eq.pct_change().dropna()
    held = (eq.index.to_series().apply(lambda d: ((tr.entry_date <= d) & (tr.exit_date > d)).any())).mean()
    return dict(trades=len(tr), win_rate=len(w) / len(tr), avg_win=w.ret.mean() if len(w) else 0,
                avg_loss=lz.ret.mean() if len(lz) else 0, payoff=(w.ret.mean() / -lz.ret.mean()) if len(w) and len(lz) else np.nan,
                avg_ret=tr.ret.mean(), pf=w.pnl.sum() / abs(lz.pnl.sum()) if len(lz) and lz.pnl.sum() else np.inf,
                days=tr.days.mean(), cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
                mdd=(eq / eq.cummax() - 1).min(), sharpe=r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan,
                best=tr.ret.max(), worst=tr.ret.min(), in_mkt=held)
