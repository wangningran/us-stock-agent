"""日线事件驱动回测。第 t 日收盘出信号 → 第 t+1 日限价单（当日有效）→ 括号单出场。"""
from __future__ import annotations

import pandas as pd

from .signals import market_ok, rank_candidates
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

    cash = float(rc["initial_equity"])
    positions: dict[str, Position] = {}
    pending: list[OrderPlan] = []
    trades, equity_curve = [], {}
    last_close: dict[str, float] = {}

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
        # 1) 昨日收盘后挂的限价单
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

        # 2) 已有持仓出场
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

        # 3) 收盘估值
        for t, df in feats.items():
            if day in df.index:
                last_close[t] = float(df.at[day, "close"])
        equity = cash + sum(p.shares * last_close.get(t, p.entry) for t, p in positions.items())
        equity_curve[day] = equity

        # 4) 生成明日挂单
        if use_mkt and not bool(mkt.get(day, False)):
            continue
        slots = rc["max_positions"] - len(positions)
        for t, row in rank_candidates(feats, day):
            if slots <= 0:
                break
            if t in positions:
                continue
            plan = plan_order(t, day, row, equity, cfg)
            if plan:
                pending.append(plan)
                slots -= 1

    # 回测结束时按最后收盘价平掉剩余持仓
    if len(days):
        for t, pos in list(positions.items()):
            close_pos(pos, last_close.get(t, pos.entry), days[-1], "end")

    return pd.DataFrame(trades), pd.Series(equity_curve, name="equity")
