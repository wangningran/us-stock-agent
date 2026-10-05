"""实盘每日报告：RSI(2) 均值回归策略 + SPY 分批建仓，维护一个"模型账户"。

假设你完全按报告执行：
- 报告在收盘前约 30 分钟生成，用当时的实时价近似收盘价
- 买卖都按收盘价成交（收盘市价单 MOC，或 12:50 前按参考价挂限价单）
- 买入后立即挂 8% 的 GTC 止损单

状态保存在 state/ 目录（提交到仓库），下次运行时用正式收盘价校正上次的近似成交价。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT
from .meanrev import add_dollar_volume_rank, build_custom, ext_features, market_context

STATE_DIR = ROOT / "state"
POS_COLS = ["ticker", "entry_date", "entry", "shares", "stop", "provisional", "confirmed"]
TRADE_COLS = ["ticker", "entry_date", "entry", "exit_date", "exit", "shares", "pnl", "ret", "reason", "provisional",
              "confirmed"]
EXTRA_LABELS = {"IBIT": "比特币 ETF"}


def extra_entry_rule(d, m):
    """比特币等额外标的：经典 RSI(2)，不要求市场压力 / MACD / 成交额排名。"""
    return (d.close > d.sma200) & (d.rsi2 < 10)


# ---------- 策略规则（与 scripts/final_rsi2.py 一致） ----------
def entry_rule(d, m):
    stress = (m.vix > 20) | (m.spy_rsi2 < 30)
    return (d.close > d.sma200) & (d.rsi2 < 10) & stress & (d.macd_hist > 0)


def exit_rule(d, m):
    return d.close > d.sma5


def rank_rule(d, m):
    return d.rsi2


# ---------- 状态 ----------
def default_account(strategy_capital: float, spy_budget: float, tranches: int = 4, weekday: int = 3) -> dict:
    return {
        "strategy_capital": strategy_capital, "cash": strategy_capital,
        "max_positions": 5, "max_new_per_day": 3, "top_n": 150, "extras": ["IBIT"],
        "stop_pct": 0.08, "time_stop": 10,
        "spy": {"ticker": "SPYM", "budget": spy_budget, "tranches": tranches, "done": 0, "weekday": weekday,
                "shares": 0, "cost": 0.0, "fills": []},
        "last_run": None,
    }


def load_state(state_dir: Path = STATE_DIR):
    acct = json.loads((state_dir / "account.json").read_text())
    pos_p, tr_p = state_dir / "positions.csv", state_dir / "trades.csv"
    pos = pd.read_csv(pos_p, parse_dates=["entry_date"]) if pos_p.exists() else pd.DataFrame(columns=POS_COLS)
    trades = pd.read_csv(tr_p, parse_dates=["entry_date", "exit_date"]) if tr_p.exists() else pd.DataFrame(columns=TRADE_COLS)
    for df in (pos, trades):
        if "confirmed" not in df:
            df["confirmed"] = False
        df["confirmed"] = df["confirmed"].fillna(False).astype(bool)
        df["provisional"] = df["provisional"].fillna(False).astype(bool)
    return acct, pos, trades


def save_state(acct, pos, trades, state_dir: Path = STATE_DIR):
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "account.json").write_text(json.dumps(acct, indent=2, ensure_ascii=False, default=str))
    pos, trades = pos.copy(), trades.copy()
    for df, cols in ((pos, ["entry_date"]), (trades, ["entry_date", "exit_date"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d")
        for c in ("entry", "exit", "stop"):
            if c in df:
                df[c] = pd.to_numeric(df[c]).round(2)
    pos.to_csv(state_dir / "positions.csv", index=False)
    trades.to_csv(state_dir / "trades.csv", index=False)


# ---------- 主流程 ----------
@dataclass
class Order:
    side: str          # BUY / SELL
    ticker: str
    shares: int
    ref_price: float
    note: str


def _official_close(feats, t, day):
    df = feats.get(t)
    if df is None or day not in df.index:
        return None
    return float(df.at[day, "close"])


def run_live(prices: dict[str, pd.DataFrame], bench: pd.DataFrame, vix: pd.DataFrame, members: pd.DataFrame | None,
             acct: dict, pos: pd.DataFrame, trades: pd.DataFrame, today: pd.Timestamp,
             core: pd.DataFrame | None = None, extras: dict[str, pd.DataFrame] | None = None):
    """返回 (report_markdown, acct, pos, trades)。today 为美东日期（normalize）。

    core：核心指数 ETF（默认 SPYM）的行情；None 时用 bench。
    extras：额外标的（如 IBIT 比特币 ETF）的行情，使用 extra_entry_rule。
    """
    core = bench if core is None else core
    day = bench.index[-1]
    if day != today:
        return (f"# {today:%Y-%m-%d} 休市或尚无今日数据\n\n最新数据日期 {day:%Y-%m-%d}，今天不操作。\n", acct, pos, trades)

    mkt = market_context(bench, vix)
    base = {t: ext_features(p) for t, p in prices.items()}
    add_dollar_volume_rank(base, bench.index, members)
    top_n = acct.get("top_n")
    entry = entry_rule if not top_n else (lambda d, m: entry_rule(d, m) & (d.dv_rank <= top_n))
    feats = build_custom(base, mkt, entry, exit_rule, rank_rule, members)
    for t, px in (extras or {}).items():
        if len(px) and px.index[-1] == day:
            feats.update(build_custom({t: ext_features(px)}, mkt, extra_entry_rule, exit_rule, rank_rule, None))
    stop_pct, time_stop, maxp = acct["stop_pct"], acct["time_stop"], acct["max_positions"]
    pos = pos.copy()
    trades = trades.copy()
    notes = []

    # 1) 用正式收盘价校正以前的近似成交价
    for i, r in pos.iterrows():
        if r.provisional and not r.confirmed and r.entry_date < day:
            px = _official_close(feats, r.ticker, r.entry_date)
            if px:
                acct["cash"] += (r.entry - px) * r.shares
                pos.at[i, "entry"], pos.at[i, "stop"] = px, round(px * (1 - stop_pct), 2)
            pos.at[i, "provisional"] = False
    for i, r in trades.iterrows():
        if r.provisional and not r.confirmed and r.exit_date < day:
            if r.reason != "stop":
                px = _official_close(feats, r.ticker, r.exit_date)
                if px:
                    acct["cash"] += (px - r.exit) * r.shares
                    trades.at[i, "exit"] = px
            entry = trades.at[i, "entry"]
            trades.at[i, "pnl"] = round((trades.at[i, "exit"] - entry) * r.shares, 2)
            trades.at[i, "ret"] = round(trades.at[i, "exit"] / entry - 1, 4)
            trades.at[i, "provisional"] = False
    spy_fills = acct["spy"]["fills"]
    for f in spy_fills:
        if f.get("provisional") and pd.Timestamp(f["date"]) < day:
            px = _official_close({"core": core}, "core", pd.Timestamp(f["date"]))
            if px:
                acct["spy"]["cost"] += (px - f["price"]) * f["shares"]
                f["price"] = px
            f["provisional"] = False

    # 2) 持仓：止损 / 卖出信号 / 时间止损
    orders: list[Order] = []
    holdings = []
    keep = []
    for i, r in pos.iterrows():
        df = feats.get(r.ticker)
        if df is None or day not in df.index:
            keep.append(i)
            holdings.append((r, None, None, None, "⚠️ 无今日数据，请手动检查"))
            continue
        bar = df.loc[day]
        held = int((df.index > r.entry_date).sum())
        last4 = float(df.close.iloc[-5:-1].mean())      # 今日收盘需高于此价 ⇔ 收盘 > 5 日线
        exit_px, reason = None, None
        if r.entry_date < day and bar.low <= r.stop:
            exit_px, reason = float(min(bar.open, r.stop)), "stop"
        elif r.entry_date < day and bar.close > last4:
            exit_px, reason = float(bar.close), "signal"
        elif held >= time_stop:
            exit_px, reason = float(bar.close), "time"
        if reason:
            prov = reason != "stop"
            acct["cash"] += exit_px * r.shares
            trades = pd.concat([trades, pd.DataFrame([{
                "ticker": r.ticker, "entry_date": r.entry_date, "entry": r.entry, "exit_date": day,
                "exit": round(exit_px, 2), "shares": r.shares, "pnl": round((exit_px - r.entry) * r.shares, 2),
                "ret": round(exit_px / r.entry - 1, 4), "reason": reason, "provisional": prov,
                "confirmed": False}])], ignore_index=True)
            if reason == "stop":
                notes.append(f"{r.ticker} 盘中已跌破止损价 ${r.stop:.2f}，止损单应已成交（如未挂止损单，请立即卖出）")
            else:
                why = "收盘价高于 5 日线" if reason == "signal" else f"已持有 {held} 天，时间止损"
                orders.append(Order("SELL", r.ticker, int(r.shares), float(bar.close),
                                    f"{why}（卖出线 ${last4:.2f}，现价 ${bar.close:.2f}）；记得撤销该股的止损单"))
        else:
            keep.append(i)
            holdings.append((r, float(bar.close), last4, held, "持有"))
    pos = pos.loc[keep].reset_index(drop=True)

    # 3) 估算权益
    def px_now(t):
        df = feats.get(t)
        return float(df.close.iloc[-1]) if df is not None else None
    equity = acct["cash"] + sum(r.shares * (px_now(r.ticker) or r.entry) for _, r in pos.iterrows())

    # 4) 新开仓
    m_today = mkt.loc[day]
    stress = bool((m_today.vix > 20) or (m_today.spy_rsi2 < 30))
    slots = maxp - len(pos)
    max_new = acct.get("max_new_per_day")
    if max_new:
        slots = min(slots, max_new)
    per_slot = equity / maxp
    held_set = set(pos.ticker) | {o.ticker for o in orders}
    cands = [(t, df.loc[day]) for t, df in feats.items()
             if day in df.index and bool(df.at[day, "entry"]) and t not in held_set]
    cands.sort(key=lambda x: x[1]["rsi2"])
    n_cands = len(cands)
    skipped = []
    for t, row in cands:
        if slots <= 0:
            break
        price = float(row.close)
        shares = math.floor(min(per_slot, acct["cash"]) / price)
        if shares < 1:
            skipped.append(f"{t}（${price:.0f}，单笔额度 ${per_slot:.0f} 买不起 1 股）")
            continue
        acct["cash"] -= shares * price
        pos = pd.concat([pos, pd.DataFrame([{
            "ticker": t, "entry_date": day, "entry": round(price, 2), "shares": shares,
            "stop": round(price * (1 - stop_pct), 2), "provisional": True, "confirmed": False}])],
            ignore_index=True)
        why = (f"{EXTRA_LABELS.get(t, '额外标的')}，RSI(2)={row.rsi2:.1f}，200 日线 ${row.sma200:.2f}"
               if t in (extras or {}) else
               f"RSI(2)={row.rsi2:.1f}，MACD 柱 {row.macd_hist:+.2f}，200 日线 ${row.sma200:.2f}，"
               f"成交额排名第 {int(row.dv_rank)}")
        orders.append(Order("BUY", t, shares, price, why))
        slots -= 1

    # 5) SPY 分批建仓
    spy = acct["spy"]
    spy_order = None
    spy_px = float(core.close.iloc[-1])
    core_t = spy.get("ticker", "SPY")
    if spy["done"] < spy["tranches"] and (day.weekday() == spy["weekday"] or spy["done"] == 0):
        remaining_budget = spy["budget"] - spy["cost"]
        amount = remaining_budget / (spy["tranches"] - spy["done"])
        sh = math.floor(amount / spy_px)
        if sh < 1 and remaining_budget >= spy_px:   # 每批预算不足 1 股时，至少买 1 股
            sh = 1
        if sh >= 1:
            spy["shares"] += sh
            spy["cost"] += sh * spy_px
            spy["done"] += 1
            spy_fills.append({"date": str(day.date()), "shares": sh, "price": spy_px, "provisional": True})
            spy_order = Order("BUY", core_t, sh, spy_px, f"标普 500 指数 ETF 分批建仓 第 {spy['done']}/{spy['tranches']} 批")
        else:
            notes.append(f"{core_t} 剩余预算 ${remaining_budget:.0f} 不够买 1 股（${spy_px:.0f}），建仓结束")
            spy["done"] += 1

    acct["last_run"] = str(day.date())
    if n_cands > len([o for o in orders if o.side == "BUY"]):
        notes.append(f"今日共 {n_cands} 只符合条件，按 RSI(2) 最低（最超卖）选出前 "
                     f"{len([o for o in orders if o.side == 'BUY'])} 只")
    report = _render(day, m_today, stress, orders, spy_order, holdings, pos, trades, acct, equity,
                     feats, notes, skipped, spy_px)
    return report, acct, pos, trades


_NAMES: dict[str, str] = {}
_EARN: dict[str, object] = {}
EARNINGS_WARN_DAYS = 10   # 与时间止损一致：预计持有期内


def _next_earnings(ticker: str, day: pd.Timestamp):
    """下一个财报日（yfinance calendar），取不到返回 None。"""
    if ticker not in _EARN:
        try:
            import yfinance as yf
            dates = (yf.Ticker(ticker).calendar or {}).get("Earnings Date") or []
            future = sorted(d for d in dates if pd.Timestamp(d) >= day)
            _EARN[ticker] = pd.Timestamp(future[0]) if future else None
        except Exception:  # noqa: BLE001
            _EARN[ticker] = None
    return _EARN[ticker]


def _earnings_text(ticker: str, day: pd.Timestamp) -> tuple[str, bool]:
    """返回（说明文字, 是否在预计持有期内）。"""
    ed = _next_earnings(ticker, day)
    if ed is None:
        return "", False
    n = int(np.busday_count(day.date(), ed.date()))
    return f"{ed:%m-%d}（{n} 个交易日后）", n <= EARNINGS_WARN_DAYS


def _name(ticker: str) -> str:
    """公司名（yfinance），取不到就返回空。"""
    if ticker not in _NAMES:
        try:
            import yfinance as yf
            info = yf.Ticker(ticker).info
            _NAMES[ticker] = info.get("shortName") or info.get("longName") or ""
        except Exception:  # noqa: BLE001
            _NAMES[ticker] = ""
    return f"（{_NAMES[ticker]}）" if _NAMES[ticker] else ""


def _render(day, m, stress, orders, spy_order, holdings, pos, trades, acct, equity, feats, notes, skipped, spy_px):
    L = [f"# 📊 RSI(2) 策略 · {day:%Y-%m-%d}（模拟盘）", ""]
    L.append(f"市场：VIX **{m.vix:.1f}**，SPY RSI(2) **{m.spy_rsi2:.0f}** → "
             + ("✅ 市场压力条件满足，股票可以开新仓" if stress else
                "⏸ 市场平静（VIX≤20 且 SPY 未超卖），股票不开新仓（比特币 ETF 不受此限制）"))
    L.append("> 价格为报告生成时的实时价，收盘可能略有不同。请在 **12:50（温哥华时间）前**下单。")
    L.append("")
    L.append("## 一、今日操作")
    todo = [o for o in orders if o.side == "SELL"] + [o for o in orders if o.side == "BUY"]
    if spy_order:
        todo.append(spy_order)
    if not todo:
        L.append("今天无需操作。")
    for i, o in enumerate(todo, 1):
        if o.side == "SELL":
            L.append(f"{i}. 🔴 **卖出 {o.ticker}{_name(o.ticker)} {o.shares} 股**，参考价 ${o.ref_price:.2f}，收盘市价单（MOC）。{o.note}")
        elif o.ticker == acct["spy"].get("ticker", "SPY"):
            L.append(f"{i}. 🟢 **买入 SPY {o.shares} 股**，参考价 ${o.ref_price:.2f}（约 ${o.shares * o.ref_price:,.0f}）。"
                     f"{o.note}，长期持有，不设止损")
        else:
            stop = o.ref_price * (1 - acct['stop_pct'])
            L.append(f"{i}. 🟢 **买入 {o.ticker}{_name(o.ticker)} {o.shares} 股**，参考价 ${o.ref_price:.2f}（约 ${o.shares * o.ref_price:,.0f}），"
                     f"收盘市价单（MOC），或限价 ${o.ref_price * 1.003:.2f}")
            L.append(f"   - 成交后挂 **GTC 止损单 ≈ ${stop:.2f}**（成交价 × 0.92）")
            L.append(f"   - 止盈：无固定价格，收盘价高于 5 日线就卖，报告每天会给出具体卖出价")
            L.append(f"   - 依据：{o.note}")
            et, soon = _earnings_text(o.ticker, day)
            if soon:
                L.append(f"   - ⚠️ **财报 {et}**，在预计持有期内，财报后容易跳空，可考虑放弃或减半")
            elif et:
                L.append(f"   - 📅 下次财报 {et}")
    for n in notes:
        L.append(f"- ⚠️ {n}")
    if skipped:
        L.append(f"- 资金不足跳过：{'；'.join(skipped)}")
    L.append("")

    L.append("## 二、持仓（买入后的第二天起检查卖出条件）")
    rows = [h for h in holdings if h[4] == "持有"]
    new = [o for o in orders if o.side == "BUY"]
    if not rows and not new:
        L.append("当前无策略持仓。")
    else:
        L.append("| 股票 | 状态 | 股数 | 成本 | 现价 | 浮盈 | 已持有 | 止损价 | 明天卖出线* |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r, px, last4, held, _ in rows:
            df = feats[r.ticker]
            nxt = float(df.close.iloc[-4:].mean())
            st = "✅ 已确认" if r.confirmed else "⏳ 未确认"
            L.append(f"| {r.ticker} | {st} | {r.shares} | ${r.entry:.2f} | ${px:.2f} | {px / r.entry - 1:+.1%} | {held} 天 "
                     f"| ${r.stop:.2f} | ${nxt:.2f} |")
        for o in new:
            df = feats[o.ticker]
            nxt = float(df.close.iloc[-4:].mean())
            L.append(f"| {o.ticker}（今日买入） | ⏳ 未确认 | {o.shares} | ${o.ref_price:.2f} | — | — | 0 天 "
                     f"| ${o.ref_price * (1 - acct['stop_pct']):.2f} | ${nxt:.2f} |")
        L.append("")
        L.append("\\* 明天收盘价高于「卖出线」就卖出（等于收盘价站上 5 日线）。最多持有 10 个交易日。")
        warns = []
        for t in [r.ticker for r, *_ in rows]:
            et, soon = _earnings_text(t, day)
            if soon:
                warns.append(f"{t} {et}")
        if warns:
            L.append(f"📅 **持仓财报提醒**：{'；'.join(warns)}。财报前后容易跳空，8% 止损可能挡不住，可考虑财报前卖出")
        if any(not r.confirmed for r, *_ in rows) or new:
            L.append("⏳ 未确认 = 按收盘价假设已成交。成交后请告诉我实际股数和价格（没买也告诉我），之后按实际记录计算。")
    L.append("")

    spy = acct["spy"]
    strat_val = equity
    L.append("## 三、模型账户")
    L.append(f"- 策略部分：权益约 **${strat_val:,.0f}**（初始 ${acct['strategy_capital']:,.0f}，"
             f"{strat_val / acct['strategy_capital'] - 1:+.1%}），现金 ${acct['cash']:,.0f}，持仓 {len(pos)} 只")
    if spy["budget"] <= 0:
        pass
    elif spy["shares"]:
        val = spy["shares"] * spy_px
        L.append(f"- 指数部分（{spy.get('ticker', 'SPY')}）：{spy['shares']} 股，成本 ${spy['cost']:,.0f}，市值约 ${val:,.0f}"
                 f"（{val / spy['cost'] - 1:+.1%}），建仓进度 {spy['done']}/{spy['tranches']}")
    else:
        L.append(f"- 指数部分（{spy.get('ticker', 'SPY')}）：预算 ${spy['budget']:,.0f}，建仓进度 {spy['done']}/{spy['tranches']}")
    closed = trades[~trades.provisional.astype(bool)] if len(trades) else trades
    if len(closed):
        L.append(f"- 已平仓 {len(closed)} 笔：胜率 {(closed.pnl > 0).mean():.0%}，累计盈亏 ${closed.pnl.sum():+,.0f}，"
                 f"平均每笔 {closed.ret.mean():+.2%}")
    L.append("")
    L.append("> 本报告由规则自动生成，仅供研究和模拟交易参考，不构成投资建议。")
    return "\n".join(L)
