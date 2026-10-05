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
    lines = [f"Trades           {s['trades']}"]
    if s["trades"]:
        lines += [
            f"Win rate         {pct(s['win_rate'])}",
            f"Avg win / loss   {pct(s['avg_win_pct'])} / {pct(s['avg_loss_pct'])}",
            f"Payoff ratio     {s['payoff_ratio']:.2f}",
            f"Expectancy (R)   {s['expectancy_r']:.3f}",
            f"Profit factor    {s['profit_factor']:.2f}",
            f"Avg days held    {s['avg_days']:.1f}",
            f"Exit reasons     {s['exit_reasons']}",
        ]
    if "total_return" in s:
        lines += [
            f"Total return     {pct(s['total_return'])}   (benchmark {pct(s.get('bench_total_return', float('nan')))})",
            f"CAGR             {pct(s['cagr'])}",
            f"Max drawdown     {pct(s['max_drawdown'])}   (benchmark {pct(s.get('bench_max_drawdown', float('nan')))})",
            f"Sharpe           {s['sharpe']:.2f}",
        ]
    return "\n".join(lines)
