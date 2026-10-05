"""短线均值回归（高胜率类策略）+ 短均线趋势策略，统一的信号式回测。

经典规则参考：
- Larry Connors RSI(2)：收盘 > 200 日线，RSI(2) < 10 买入；收盘 > 5 日线卖出
- 累积 RSI(2)：两日 RSI(2) 之和 < 35 买入；RSI(2) > 65 卖出
- IBS（收盘在当日高低区间的位置）< 0.2 且收跌买入；收盘高于前一日最高价卖出
- EMA6/EMA12 金叉买入、死叉卖出（短均线趋势跟踪）

执行方式：
- close：信号日收盘成交（需在收盘前约 15 分钟按近似价格下市价收盘单 MOC）
- open ：信号日次日开盘成交（看盘前报告下单）
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


# 每条规则：entry(df) / exit(df) 返回 bool Series（收盘后判断），rank(df) 越小越优先
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
    "ema_trend_rsi2_dip": dict(   # 短期上升趋势中的急跌
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
                    market_filter: bool = False, start=None, end=None):
    """信号式回测。等权：每笔 = 权益 / max_positions。"""
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
        # 次日开盘执行（open 模式）
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
        # 盘中止损
        if stop_pct:
            for t in list(pos):
                p = pos[t]
                stop = p["px"] * (1 - stop_pct)
                if pd.notna(l[t]) and l[t] <= stop and p["d"] != day:
                    sell(t, min(o[t], stop), day, "stop")
        # 收盘：更新持有天数，判断出场
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
        # 收盘：新信号
        if market_filter and not bool(mkt.get(day, False)):
            continue
        slots = max_positions - len(pos) + (len(pend_sell) if mode == "open" else 0)
        if slots <= 0:
            continue
        cands = ENTRY.columns[ENTRY.loc[day].values]
        cands = [t for t in cands if t not in pos]
        cands.sort(key=lambda t: (pd.isna(P["rank"].at[day, t]), P["rank"].at[day, t]))
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
