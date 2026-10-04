"""交易规则：挂单计划 + 持仓出场判断。回测和每日报告共用同一套规则，保证报告 = 回测过的策略。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class OrderPlan:
    ticker: str
    signal_date: pd.Timestamp
    limit: float
    atr: float
    shares: int
    stop: float          # 按限价估算，成交后按实际成交价重算
    target: float
    analyst_score: float
    mom: float
    rsi: float


@dataclass
class Position:
    ticker: str
    entry_date: pd.Timestamp
    entry: float
    shares: int
    stop: float
    target: float
    initial_risk: float  # 每股 1R
    signal_date: pd.Timestamp | None = None
    days_held: int = 0
    meta: dict = field(default_factory=dict)


def plan_order(ticker: str, day: pd.Timestamp, row: pd.Series, equity: float, cfg: dict) -> OrderPlan | None:
    tc, rc = cfg["trade"], cfg["risk"]
    atr = float(row["atr"])
    limit = round(float(row["close"]) - tc["entry_atr_mult"] * atr, 2)
    stop_dist = tc["stop_atr_mult"] * atr
    if limit <= 0 or stop_dist <= 0:
        return None
    shares = math.floor(min(equity * rc["risk_per_trade"] / stop_dist,
                            equity * rc["max_position_pct"] / limit))
    if shares < 1:
        return None
    return OrderPlan(
        ticker=ticker, signal_date=day, limit=limit, atr=atr, shares=shares,
        stop=round(limit - stop_dist, 2), target=round(limit + tc["target_atr_mult"] * atr, 2),
        analyst_score=float(row["analyst_score"]), mom=float(row["mom"]), rsi=float(row["rsi"]),
    )


def try_fill(order: OrderPlan, bar: pd.Series) -> float | None:
    """当日限价买单是否成交。要求最低价触及限价；低开则按开盘价成交。"""
    if bar["low"] <= order.limit:
        return float(min(bar["open"], order.limit))
    return None


def open_position(order: OrderPlan, fill: float, day: pd.Timestamp, cfg: dict) -> Position:
    tc = cfg["trade"]
    risk = tc["stop_atr_mult"] * order.atr
    return Position(
        ticker=order.ticker, entry_date=day, entry=fill, shares=order.shares,
        stop=fill - risk, target=fill + tc["target_atr_mult"] * order.atr,
        initial_risk=risk, signal_date=order.signal_date,
    )


def check_exit(pos: Position, bar: pd.Series, cfg: dict, fill_day: bool = False) -> tuple[float, str] | None:
    """用日线判断出场。同一根K线同时触及止损和止盈时，保守地按止损处理。

    fill_day=True：开仓当天只检查止损（无法确定盘中先后顺序，不计入止盈）。
    """
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]
    if not fill_day:
        if o <= pos.stop:
            return float(o), "stop_gap"
        if o >= pos.target:
            return float(o), "target_gap"
    if l <= pos.stop:
        return float(pos.stop), "breakeven" if pos.stop >= pos.entry else "stop"
    if not fill_day and h >= pos.target:
        return float(pos.target), "target"
    if pos.days_held >= cfg["trade"]["max_hold_days"]:
        return float(c), "time"
    return None


def update_trailing(pos: Position, bar: pd.Series, cfg: dict) -> None:
    """收盘后更新：浮盈达到 breakeven_at_r 后，止损上移到成本价（次日起生效）。"""
    be = cfg["trade"].get("breakeven_at_r")
    if be and bar["high"] >= pos.entry + be * pos.initial_risk:
        pos.stop = max(pos.stop, pos.entry)
