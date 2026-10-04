"""每日报告：盘前操作清单（Markdown），供手动在 moomoo 下单。"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .signals import market_ok, rank_candidates
from .strategy import plan_order

POSITION_COLS = ["ticker", "entry_date", "entry", "shares", "stop", "target"]


def load_positions(path: str | Path) -> pd.DataFrame:
    """持仓文件由你手动维护（成交后填一行，平仓后删掉）。"""
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
            pt = f"，目标价 {r.pt_prior:.0f} → {r.pt_current:.0f}" if pd.notna(r.pt_prior) and r.pt_prior else f"，目标价 {r.pt_current:.0f}"
        lines.append(f"{ts:%m-%d} {r.firm}：{r.action} {grade}{pt}")
    return lines


def build_report(feats: dict[str, pd.DataFrame], events: dict[str, pd.DataFrame], bench: pd.DataFrame,
                 cfg: dict, positions: pd.DataFrame, equity: float, check_earnings: bool = True) -> str:
    tc, rc, sc = cfg["trade"], cfg["risk"], cfg["signals"]
    day = bench.index[-1]
    b = bench["close"]
    mkt = bool(market_ok(bench).iloc[-1])
    lookback_start = bench.index[max(0, len(bench) - sc["analyst_lookback_days"])]

    out = [f"# 交易报告 · 数据截至 {day:%Y-%m-%d} 收盘", ""]
    out.append(f"**大盘状态**：{'✅ 可开新仓' if mkt or not sc.get('market_filter', True) else '⛔ 暂停开新仓'}"
               f" — {cfg['benchmark']} {b.iloc[-1]:.2f}，200 日线 {b.rolling(200).mean().iloc[-1]:.2f}")
    out.append(f"**账户权益（估）**：${equity:,.0f}　每笔风险 {rc['risk_per_trade']:.1%}　最多持仓 {rc['max_positions']} 只")
    out.append("")

    # ---- 持仓管理 ----
    out.append("## 一、持仓管理")
    actions, held = [], set()
    if positions.empty:
        out.append("当前无持仓。")
    else:
        out.append("| 股票 | 成本 | 现价 | 浮盈 | R | 持有天数 | 止损 / 止盈 | 操作 |")
        out.append("|---|---|---|---|---|---|---|---|")
        for _, p in positions.iterrows():
            t = p.ticker
            held.add(t)
            df = feats.get(t)
            if df is None:
                out.append(f"| {t} | {p.entry:.2f} | — | — | — | — | {p.stop:.2f} / {p.target:.2f} | 不在股票池，请手动管理 |")
                continue
            since = df[df.index > p.entry_date]
            last = df.iloc[-1]
            risk = p.entry - p.stop if p.stop < p.entry else tc["stop_atr_mult"] * last.atr
            r_now = (last.close - p.entry) / risk if risk > 0 else 0
            days_held = len(since)
            todo = "持有"
            if len(since) and since.low.min() <= p.stop:
                todo = "⚠️ 已触及止损价，确认是否已卖出"
            elif len(since) and since.high.max() >= p.target:
                todo = "🎯 已触及止盈价，确认是否已卖出"
            elif days_held >= tc["max_hold_days"]:
                todo = "⏰ 时间止损：开盘卖出"
            elif p.stop < p.entry and len(since) and since.high.max() >= p.entry + tc["breakeven_at_r"] * risk:
                todo = f"⬆️ 止损上移到成本价 {p.entry:.2f}"
            if todo != "持有":
                actions.append(f"{t}：{todo}")
            out.append(f"| {t} | {p.entry:.2f} | {last.close:.2f} | {last.close / p.entry - 1:+.1%} | {r_now:+.2f} "
                       f"| {days_held} | {p.stop:.2f} / {p.target:.2f} | {todo} |")
    out.append("")

    # ---- 新开仓 ----
    out.append("## 二、今日新开仓挂单")
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
        out.append("今日无新信号。")
    for i, pl in enumerate(plans, 1):
        last = feats[pl.ticker].iloc[-1]
        out.append(f"### {i}. {pl.ticker}　限价买入 ${pl.limit:.2f} × {pl.shares} 股"
                   f"（约 ${pl.limit * pl.shares:,.0f}，{pl.limit * pl.shares / equity:.1%} 仓位）")
        out.append(f"- **附加止损** ${pl.stop:.2f}（{pl.stop / pl.limit - 1:+.1%}）　"
                   f"**附加止盈** ${pl.target:.2f}（{pl.target / pl.limit - 1:+.1%}）　当日有效，未成交就放弃")
        out.append(f"- 现价 {last.close:.2f}，ATR {pl.atr:.2f}，3 个月动量 {pl.mom:+.1%}，RSI {pl.rsi:.0f}，"
                   f"站上 50/200 日线")
        out.append(f"- 分析师分数 {pl.analyst_score:.0f}（近 {sc['analyst_lookback_days']} 个交易日）：")
        for line in _recent_events(events.get(pl.ticker), lookback_start):
            out.append(f"  - {line}")
        if check_earnings:
            ed = next_earnings(pl.ticker)
            if ed and (ed - day.date()).days <= tc["max_hold_days"] * 7 // 5:
                out.append(f"- ⚠️ 财报日 {ed}，在预计持有期内，跳空风险大，可考虑放弃或减半")
        out.append("")

    # ---- 汇总 ----
    out.append("## 三、今日操作清单")
    todo = actions + [f"{p.ticker}：限价 ${p.limit:.2f} 买入 {p.shares} 股，附止损 ${p.stop:.2f} / 止盈 ${p.target:.2f}"
                      for p in plans]
    out += [f"{i}. {x}" for i, x in enumerate(todo, 1)] if todo else ["无需操作。"]
    out.append("")
    out.append("> 成交后请把实际成交价写进 `positions.csv`，止损/止盈按实际成交价重算："
               f"止损 = 成交价 − {tc['stop_atr_mult']}×ATR，止盈 = 成交价 + {tc['target_atr_mult']}×ATR。")
    out.append("> 本报告由规则自动生成，仅供研究参考，不构成投资建议。")
    return "\n".join(out)
