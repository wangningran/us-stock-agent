"""Daily-bar event-driven backtest: signal at close of day t -> day-only limit order on t+1 -> bracket exits."""
from __future__ import annotations

import pandas as pd

from .signals import market_ok
from .strategy import OrderPlan, Position, check_exit, open_position, plan_order, try_fill, update_trailing


def run_backtest(feats: dict[str, pd.DataFrame], bench: pd.DataFrame, cfg: dict,
                 start: str | None = None, end: str | None = None) -> tuple[pd.DataFrame, pd.Series]:
    rc = cfg["risk"]
    slip = rc["slippage_bps"] / 1e4
    fee = rc["commission_per_order"]
    use_mkt = cfg["signals"].get("market_filter", True)

    days = bench.index
    if start:
        days = days[days >= pd.Timestamp(start)]
    if end:
        days = days[days <= pd.Timestamp(end)]
    mkt = market_ok(bench)

    # Pre-align wide panels so each day does not loop over every ticker
    tickers = list(feats)
    close_px = pd.DataFrame({t: feats[t]["close"] for t in tickers}).reindex(days).ffill()
    cand = pd.DataFrame({t: feats[t]["candidate"] for t in tickers}).reindex(days).fillna(False).astype(bool)
    score = pd.DataFrame({t: feats[t]["analyst_score"] for t in tickers}).reindex(days)
    mom = pd.DataFrame({t: feats[t]["mom"] for t in tickers}).reindex(days)

    cash = float(rc["initial_equity"])
    positions: dict[str, Position] = {}
    pending: list[OrderPlan] = []
    trades, equity_curve = [], {}

    def close_pos(pos: Position, price: float, day, reason: str):
        nonlocal cash
        px = price * (1 - slip)
        cash += pos.shares * px - fee
        pnl = (px - pos.entry) * pos.shares - 2 * fee
        trades.append({
            "ticker": pos.ticker, "signal_date": pos.signal_date, "entry_date": pos.entry_date,
            "entry": round(pos.entry, 4), "exit_date": day, "exit": round(px, 4),
            "shares": pos.shares, "pnl": round(pnl, 2),
            "r": round((px - pos.entry) / pos.initial_risk, 3),
            "ret": round(px / pos.entry - 1, 5), "days": pos.days_held, "reason": reason,
        })

    for day in days:
        # 1) Limit orders placed after yesterday's close
        for order in pending:
            df = feats[order.ticker]
            if day not in df.index or order.ticker in positions:
                continue
            bar = df.loc[day]
            fill = try_fill(order, bar)
            if fill is None:
                continue
            fill *= 1 + slip
            cost = fill * order.shares + fee
            if cost > cash:
                continue
            cash -= cost
            pos = open_position(order, fill, day, cfg)
            ex = check_exit(pos, bar, cfg, fill_day=True)
            if ex:
                close_pos(pos, ex[0], day, ex[1])
            else:
                update_trailing(pos, bar, cfg)
                positions[pos.ticker] = pos
                pos.meta["new_today"] = True
        pending = []

        # 2) Exits for existing positions
        for t in list(positions):
            pos = positions[t]
            if pos.meta.pop("new_today", False):
                continue
            df = feats[t]
            if day not in df.index:
                continue
            bar = df.loc[day]
            pos.days_held += 1
            ex = check_exit(pos, bar, cfg)
            if ex:
                close_pos(pos, ex[0], day, ex[1])
                del positions[t]
            else:
                update_trailing(pos, bar, cfg)

        # 3) Mark to market at the close
        closes = close_px.loc[day]
        equity = cash + sum(p.shares * (closes[t] if pd.notna(closes[t]) else p.entry)
                            for t, p in positions.items())
        equity_curve[day] = equity

        # 4) Generate orders for tomorrow
        if use_mkt and not bool(mkt.get(day, False)):
            continue
        slots = rc["max_positions"] - len(positions)
        if slots <= 0:
            continue
        today = cand.columns[cand.loc[day].values]
        ranked = sorted(today, key=lambda t: (score.at[day, t], mom.at[day, t]), reverse=True)
        for t in ranked:
            if slots <= 0:
                break
            if t in positions:
                continue
            plan = plan_order(t, day, feats[t].loc[day], equity, cfg)
            if plan:
                pending.append(plan)
                slots -= 1

    # Close remaining positions at the last close
    if len(days):
        last = close_px.iloc[-1]
        for t, pos in list(positions.items()):
            close_pos(pos, last[t] if pd.notna(last[t]) else pos.entry, days[-1], "end")

    return pd.DataFrame(trades), pd.Series(equity_curve, name="equity")
