from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(trades: pd.DataFrame, equity: pd.Series, bench: pd.DataFrame | None = None) -> dict:
    out: dict = {"trades": int(len(trades))}
    if len(trades):
        wins, losses = trades[trades.pnl > 0], trades[trades.pnl <= 0]
        avg_win = wins.ret.mean() if len(wins) else 0.0
        avg_loss = losses.ret.mean() if len(losses) else 0.0
        out.update({
            "win_rate": len(wins) / len(trades),
            "avg_win_pct": avg_win,
            "avg_loss_pct": avg_loss,
            "payoff_ratio": abs(avg_win / avg_loss) if avg_loss else np.inf,
            "expectancy_r": trades.r.mean(),
            "profit_factor": wins.pnl.sum() / abs(losses.pnl.sum()) if losses.pnl.sum() else np.inf,
            "avg_days": trades.days.mean(),
            "exit_reasons": trades.reason.value_counts().to_dict(),
        })
    if len(equity) > 1:
        rets = equity.pct_change().dropna()
        years = (equity.index[-1] - equity.index[0]).days / 365.25
        out.update({
            "total_return": equity.iloc[-1] / equity.iloc[0] - 1,
            "cagr": (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan,
            "max_drawdown": (equity / equity.cummax() - 1).min(),
            "sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() else np.nan,
        })
        if bench is not None:
            b = bench["close"].reindex(equity.index).ffill()
            out["bench_total_return"] = b.iloc[-1] / b.iloc[0] - 1
            out["bench_max_drawdown"] = (b / b.cummax() - 1).min()
    return out


def format_summary(s: dict) -> str:
    pct = lambda x: f"{x * 100:.1f}%"  # noqa: E731
    lines = [f"交易笔数        {s['trades']}"]
    if s["trades"]:
        lines += [
            f"胜率            {pct(s['win_rate'])}",
            f"平均盈利/亏损    {pct(s['avg_win_pct'])} / {pct(s['avg_loss_pct'])}",
            f"盈亏比          {s['payoff_ratio']:.2f}",
            f"期望值(每笔R)    {s['expectancy_r']:.3f}",
            f"利润因子        {s['profit_factor']:.2f}",
            f"平均持有天数     {s['avg_days']:.1f}",
            f"出场原因        {s['exit_reasons']}",
        ]
    if "total_return" in s:
        lines += [
            f"总收益          {pct(s['total_return'])}   (基准 {pct(s.get('bench_total_return', float('nan')))})",
            f"年化收益        {pct(s['cagr'])}",
            f"最大回撤        {pct(s['max_drawdown'])}   (基准 {pct(s.get('bench_max_drawdown', float('nan')))})",
            f"夏普比率        {s['sharpe']:.2f}",
        ]
    return "\n".join(lines)
