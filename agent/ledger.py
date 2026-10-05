"""Trade ledger: update the model account with the user's actual fills.

Recommended trades are first recorded as unconfirmed (assumed filled at the close). Once the user reports the
actual fill, these functions mark it confirmed, and later exit advice, stops and P&L use the real numbers.
"""
from __future__ import annotations

import pandas as pd


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)


def _row(df: pd.DataFrame, ticker: str) -> int | None:
    idx = df.index[df.ticker == ticker]
    return idx[-1] if len(idx) else None


def confirm_buy(acct, pos, ticker, shares, price, date=None):
    """Confirm a buy (or record one the report did not suggest). An unconfirmed lot of the same ticker is updated."""
    date = pd.Timestamp(date) if date else _today()
    unconfirmed = pos.index[(pos.ticker == ticker) & ~pos.confirmed.astype(bool)]
    i = unconfirmed[-1] if len(unconfirmed) else None   # confirmed lots are never overwritten; a new lot is added
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
    """The report recommended a buy but the user did not buy."""
    i = _row(pos, ticker)
    if i is None:
        raise ValueError(f"no position in {ticker}")
    r = pos.loc[i]
    acct["cash"] += r.entry * r.shares
    return acct, pos.drop(index=i).reset_index(drop=True)


def confirm_sell(acct, pos, trades, ticker, shares, price, date=None):
    """Confirm a sell. Matches a pending (unconfirmed) report sell first, else sells from holdings (partial ok)."""
    date = pd.Timestamp(date) if date else _today()
    pending = trades.index[(trades.ticker == ticker) & ~trades.confirmed.astype(bool)]
    if len(pending):
        j = pending[-1]
        t = trades.loc[j]
        acct["cash"] += price * shares - t.exit * t.shares
        if shares < t.shares:  # partial sell: the remainder goes back to holdings
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
        raise ValueError(f"{ticker} is neither held nor pending sale")
    r = pos.loc[i]
    if shares > r.shares:
        raise ValueError(f"only {r.shares} shares of {ticker} held")
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
    """The report recommended a sell but the user kept the position."""
    pending = trades.index[(trades.ticker == ticker) & ~trades.confirmed.astype(bool)]
    if not len(pending):
        raise ValueError(f"no pending sale for {ticker}")
    j = pending[-1]
    t = trades.loc[j]
    acct["cash"] -= t.exit * t.shares
    pos = pd.concat([pos, pd.DataFrame([{
        "ticker": ticker, "entry_date": t.entry_date, "entry": t.entry, "shares": t.shares,
        "stop": round(t.entry * (1 - acct["stop_pct"]), 2), "provisional": False, "confirmed": True}])],
        ignore_index=True)
    return acct, pos, trades.drop(index=j).reset_index(drop=True)
