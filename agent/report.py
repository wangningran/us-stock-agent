"""Legacy v1 daily report (analyst + momentum strategy): pre-market order list in Markdown."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .signals import market_ok, rank_candidates
from .strategy import plan_order

POSITION_COLS = ["ticker", "entry_date", "entry", "shares", "stop", "target"]


def load_positions(path: str | Path) -> pd.DataFrame:
    """Positions file maintained by hand (add a row after a fill, delete it after closing)."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=POSITION_COLS)
    df = pd.read_csv(path, parse_dates=["entry_date"])
    missing = set(POSITION_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"positions file missing columns: {missing}")
    return df


def next_earnings(ticker: str) -> date | None:
    try:
        import yfinance as yf
        cal = yf.Ticker(ticker).calendar or {}
        dates = cal.get("Earnings Date") or []
        future = [d for d in dates if d >= date.today()]
        return min(future) if future else None
    except Exception:  # noqa: BLE001
        return None


def _recent_events(events: pd.DataFrame | None, since: pd.Timestamp) -> list[str]:
    if events is None or events.empty:
        return []
    ev = events[events.index.normalize() >= since]
    lines = []
    for ts, r in ev.iterrows():
        grade = f"{r.from_grade or '—'} → {r.to_grade}" if pd.notna(r.from_grade) and r.from_grade else f"{r.to_grade}"
        pt = ""
        if pd.notna(r.pt_current) and r.pt_current:
            pt = f", PT {r.pt_prior:.0f} -> {r.pt_current:.0f}" if pd.notna(r.pt_prior) and r.pt_prior else f", PT {r.pt_current:.0f}"
        lines.append(f"{ts:%m-%d} {r.firm}: {r.action} {grade}{pt}")
    return lines


def build_report(feats: dict[str, pd.DataFrame], events: dict[str, pd.DataFrame], bench: pd.DataFrame,
                 cfg: dict, positions: pd.DataFrame, equity: float, check_earnings: bool = True) -> str:
    tc, rc, sc = cfg["trade"], cfg["risk"], cfg["signals"]
    day = bench.index[-1]
    b = bench["close"]
    mkt = bool(market_ok(bench).iloc[-1])
    lookback_start = bench.index[max(0, len(bench) - sc["analyst_lookback_days"])]

    out = [f"# Trading report - data as of {day:%Y-%m-%d} close", ""]
    out.append(f"**Market**: {'✅ new entries allowed' if mkt or not sc.get('market_filter', True) else '⛔ no new entries'}"
               f" - {cfg['benchmark']} {b.iloc[-1]:.2f}, 200-day SMA {b.rolling(200).mean().iloc[-1]:.2f}")
    out.append(f"**Equity (est.)**: ${equity:,.0f} | risk per trade {rc['risk_per_trade']:.1%} | max positions {rc['max_positions']}")
    out.append("")

    # ---- Holdings ----
    out.append("## 1. Holdings")
    actions, held = [], set()
    if positions.empty:
        out.append("No positions.")
    else:
        out.append("| Ticker | Cost | Price | P&L | R | Days | Stop / Target | Action |")
        out.append("|---|---|---|---|---|---|---|---|")
        for _, p in positions.iterrows():
            t = p.ticker
            held.add(t)
            df = feats.get(t)
            if df is None:
                out.append(f"| {t} | {p.entry:.2f} | — | — | — | — | {p.stop:.2f} / {p.target:.2f} | not in universe, manage manually |")
                continue
            since = df[df.index > p.entry_date]
            last = df.iloc[-1]
            risk = p.entry - p.stop if p.stop < p.entry else tc["stop_atr_mult"] * last.atr
            r_now = (last.close - p.entry) / risk if risk > 0 else 0
            days_held = len(since)
            todo = "hold"
            if len(since) and since.low.min() <= p.stop:
                todo = "⚠️ stop touched - confirm it was sold"
            elif len(since) and since.high.max() >= p.target:
                todo = "🎯 target touched - confirm it was sold"
            elif days_held >= tc["max_hold_days"]:
                todo = "⏰ time stop: sell at the open"
            elif p.stop < p.entry and len(since) and since.high.max() >= p.entry + tc["breakeven_at_r"] * risk:
                todo = f"⬆️ move stop to entry {p.entry:.2f}"
            if todo != "hold":
                actions.append(f"{t}: {todo}")
            out.append(f"| {t} | {p.entry:.2f} | {last.close:.2f} | {last.close / p.entry - 1:+.1%} | {r_now:+.2f} "
                       f"| {days_held} | {p.stop:.2f} / {p.target:.2f} | {todo} |")
    out.append("")

    # ---- New entries ----
    out.append("## 2. New limit orders")
    plans = []
    if mkt or not sc.get("market_filter", True):
        slots = rc["max_positions"] - len(held)
        for t, row in rank_candidates(feats, day):
            if slots <= 0:
                break
            if t in held:
                continue
            plan = plan_order(t, day, row, equity, cfg)
            if plan:
                plans.append(plan)
                slots -= 1
    if not plans:
        out.append("No new signals today.")
    for i, pl in enumerate(plans, 1):
        last = feats[pl.ticker].iloc[-1]
        out.append(f"### {i}. {pl.ticker} - limit buy ${pl.limit:.2f} x {pl.shares}"
                   f" (~${pl.limit * pl.shares:,.0f}, {pl.limit * pl.shares / equity:.1%} of equity)")
        out.append(f"- **Stop** ${pl.stop:.2f} ({pl.stop / pl.limit - 1:+.1%}) | "
                   f"**Target** ${pl.target:.2f} ({pl.target / pl.limit - 1:+.1%}) | day order, skip if unfilled")
        out.append(f"- Price {last.close:.2f}, ATR {pl.atr:.2f}, 3-month momentum {pl.mom:+.1%}, RSI {pl.rsi:.0f}, "
                   f"above 50/200-day SMAs")
        out.append(f"- Analyst score {pl.analyst_score:.0f} (last {sc['analyst_lookback_days']} trading days):")
        for line in _recent_events(events.get(pl.ticker), lookback_start):
            out.append(f"  - {line}")
        if check_earnings:
            ed = next_earnings(pl.ticker)
            if ed and (ed - day.date()).days <= tc["max_hold_days"] * 7 // 5:
                out.append(f"- ⚠️ earnings on {ed}, within the holding period - gap risk; consider skipping or halving")
        out.append("")

    # ---- Summary ----
    out.append("## 3. Today's actions")
    todo = actions + [f"{p.ticker}: limit buy {p.shares} @ ${p.limit:.2f}, stop ${p.stop:.2f} / target ${p.target:.2f}"
                      for p in plans]
    out += [f"{i}. {x}" for i, x in enumerate(todo, 1)] if todo else ["No action needed."]
    out.append("")
    out.append("> After a fill, record the actual price in `positions.csv` and recompute: "
               f"stop = fill - {tc['stop_atr_mult']} x ATR, target = fill + {tc['target_atr_mult']} x ATR.")
    out.append("> Generated by rules for research only; not investment advice.")
    return "\n".join(out)
