"""买卖记录：按用户实际成交更新模型账户。

报告推荐的买卖先记为"未确认"（按收盘价假设成交）；用户告知实际成交后，用这里的函数改成"已确认"，
之后的卖出建议、止损价、盈亏统计都基于实际记录。
"""
from __future__ import annotations

import pandas as pd


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def _row(df: pd.DataFrame, ticker: str) -> int | None:
    idx = df.index[df.ticker == ticker]
    return idx[-1] if len(idx) else None


def confirm_buy(acct, pos, ticker, shares, price, date=None):
    """确认买入（或记录一笔报告外的买入）。已有同名未确认持仓则改成实际股数和价格。"""
    date = pd.Timestamp(date) if date else _today()
    i = _row(pos, ticker)
    stop = round(price * (1 - acct["stop_pct"]), 2)
    if i is not None:
        r = pos.loc[i]
        acct["cash"] += r.entry * r.shares - price * shares
        pos.loc[i, ["entry_date", "entry", "shares", "stop", "provisional", "confirmed"]] = \
            [date, price, shares, stop, False, True]
    else:
        acct["cash"] -= price * shares
        pos = pd.concat([pos, pd.DataFrame([{"ticker": ticker, "entry_date": date, "entry": price, "shares": shares,
                                             "stop": stop, "provisional": False, "confirmed": True}])],
                        ignore_index=True)
    return acct, pos


def cancel_buy(acct, pos, ticker):
    """报告推荐了但实际没买。"""
    i = _row(pos, ticker)
    if i is None:
        raise ValueError(f"持仓中没有 {ticker}")
    r = pos.loc[i]
    acct["cash"] += r.entry * r.shares
    return acct, pos.drop(index=i).reset_index(drop=True)


def confirm_sell(acct, pos, trades, ticker, shares, price, date=None):
    """确认卖出。优先匹配报告记下的未确认卖出；否则从持仓中卖出（支持部分卖出）。"""
    date = pd.Timestamp(date) if date else _today()
    pending = trades.index[(trades.ticker == ticker) & ~trades.confirmed.astype(bool)]
    if len(pending):
        j = pending[-1]
        t = trades.loc[j]
        acct["cash"] += price * shares - t.exit * t.shares
        if shares < t.shares:  # 只卖了一部分：剩余放回持仓
            pos = pd.concat([pos, pd.DataFrame([{
                "ticker": ticker, "entry_date": t.entry_date, "entry": t.entry, "shares": t.shares - shares,
                "stop": round(t.entry * (1 - acct["stop_pct"]), 2), "provisional": False, "confirmed": True}])],
                ignore_index=True)
        trades.loc[j, ["exit_date", "exit", "shares", "provisional", "confirmed"]] = [date, price, shares, False, True]
        trades.loc[j, "pnl"] = round((price - t.entry) * shares, 2)
        trades.loc[j, "ret"] = round(price / t.entry - 1, 4)
        return acct, pos, trades
    i = _row(pos, ticker)
    if i is None:
        raise ValueError(f"持仓和待确认卖出中都没有 {ticker}")
    r = pos.loc[i]
    if shares > r.shares:
        raise ValueError(f"{ticker} 只持有 {r.shares} 股")
    acct["cash"] += price * shares
    trades = pd.concat([trades, pd.DataFrame([{
        "ticker": ticker, "entry_date": r.entry_date, "entry": r.entry, "exit_date": date, "exit": price,
        "shares": shares, "pnl": round((price - r.entry) * shares, 2), "ret": round(price / r.entry - 1, 4),
        "reason": "manual", "provisional": False, "confirmed": True}])], ignore_index=True)
    if shares == r.shares:
        pos = pos.drop(index=i).reset_index(drop=True)
    else:
        pos.loc[i, "shares"] = r.shares - shares
    return acct, pos, trades


def cancel_sell(acct, pos, trades, ticker):
    """报告建议卖出但实际没卖：放回持仓。"""
    pending = trades.index[(trades.ticker == ticker) & ~trades.confirmed.astype(bool)]
    if not len(pending):
        raise ValueError(f"没有 {ticker} 的待确认卖出")
    j = pending[-1]
    t = trades.loc[j]
    acct["cash"] -= t.exit * t.shares
    pos = pd.concat([pos, pd.DataFrame([{
        "ticker": ticker, "entry_date": t.entry_date, "entry": t.entry, "shares": t.shares,
        "stop": round(t.entry * (1 - acct["stop_pct"]), 2), "provisional": False, "confirmed": True}])],
        ignore_index=True)
    return acct, pos, trades.drop(index=j).reset_index(drop=True)
