"""Reconcile Nautilus Trader results with our backtester and write research/nautilus/RESULTS.md.

  python research/nautilus/compare.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "out"


def stats(tr, eq, s, e):
    eq = eq[(eq.index >= s) & (eq.index < e)]
    tr = tr[(tr.exit_date >= s) & (tr.exit_date < e)]
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    r = eq.pct_change().dropna()
    return dict(trades=len(tr), win=(tr.pnl > 0).mean(), cagr=(eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1,
                mdd=(eq / eq.cummax() - 1).min(), sharpe=r.mean() / r.std() * np.sqrt(252))


def main():
    r = pd.read_csv(OUT / "ref_trades.csv", parse_dates=["entry_date", "exit_date"])
    n = pd.read_csv(OUT / "nautilus_trades.csv", parse_dates=["entry_date", "exit_date"])
    re = pd.read_csv(OUT / "ref_equity.csv", parse_dates=["date"], index_col="date").iloc[:, 0]
    ne = pd.read_csv(OUT / "nautilus_equity.csv", parse_dates=["date"], index_col="date").iloc[:, 0]
    bars = pd.read_csv(OUT / "bars_signals.csv", parse_dates=["date"]).set_index(["ticker", "date"])

    m = r.merge(n, on=["ticker", "entry_date"], how="outer", suffixes=("_r", "_n"), indicator=True)
    both = m[m._merge == "both"]
    same_entry = (both.entry_r - both.entry_n).abs().max() < 1e-6
    same_exit_day = (both.exit_date_r == both.exit_date_n).all()
    same_reason = (both.reason_r == both.reason_n).all()
    px_diff = both[(both.exit_r - both.exit_n).abs() > 1e-3]
    gap = all(abs(x - bars.loc[(t, d), "open"]) < 1e-6 and bars.loc[(t, d), "open"] < st
              for t, d, x, st in zip(px_diff.ticker, px_diff.exit_date_r, px_diff.exit_r, px_diff.exit_n + 1e-6))
    first_gap = px_diff.exit_date_r.min()
    pre = pd.concat([re, ne], axis=1, keys=["r", "n"]).dropna()
    pre = pre[pre.index < first_gap]
    max_pre = (pre.r / pre.n - 1).abs().max()

    L = ["# Nautilus Trader cross-check: live RSI(2) configuration", "",
         "Same data (Yahoo daily bars rounded to 4 dp, point-in-time S&P 500), same signals, US$5,260, whole shares,",
         "zero commission and zero slippage in both engines, 2019-01-02 to the latest bar.", "",
         "## Trade-by-trade reconciliation", "",
         f"- Trades: ours {len(r)}, Nautilus {len(n)}; matched on ticker + entry date: {len(both)}, "
         f"unmatched: {int((m._merge != 'both').sum())}",
         f"- Entry prices identical: {'yes' if same_entry else 'NO'}; exit dates identical: "
         f"{'yes' if same_exit_day else 'NO'}; exit reasons identical: {'yes' if same_reason else 'NO'}",
         f"- Exit prices differ on {len(px_diff)} trades, all stop exits on a gap-down open "
         f"({'confirmed' if gap else 'NOT all gaps'}): we fill at the open (below the stop), Nautilus's bar "
         "execution fills at the stop price. Ours is the conservative / realistic assumption.",
         f"- Equity curves before the first gap stop ({first_gap.date()}): max difference {max_pre:.4%} "
         "(Nautilus keeps cash in cents).", "",
         "## Results", "",
         "| Period | Engine | Trades | Win rate | CAGR | Max DD | Sharpe |", "|---|---|---|---|---|---|---|"]
    for name, (s, e) in {"2019-2024": ("2019-01-01", "2025-01-01"), "2025-2026": ("2025-01-01", "2027-01-01"),
                         "2019-2026": ("2019-01-01", "2027-01-01")}.items():
        for eng, tr, eq in (("ours", r, re), ("Nautilus", n, ne)):
            x = stats(tr, eq, s, e)
            L.append(f"| {name} | {eng} | {x['trades']} | {x['win']:.1%} | {x['cagr']:+.1%} | {x['mdd']:.1%} "
                     f"| {x['sharpe']:.2f} |")
    L += ["", "## Notes", "",
          "- Both engines assume the signal is computed at the close and the order fills at that close "
          "(market-on-close). Nautilus fills a market order sent right after a daily bar at that bar's close, so "
          "it does not remove this assumption; in live trading the signal uses prices ~45 minutes before the "
          "close.",
          "- Nautilus's cash-account risk engine denies sell stops when free cash is low unless they are "
          "reduce-only; the run uses reduce-only exits.",
          "- Data-level biases (missing delisted tickers, adjusted prices) are the same in both engines."]
    (Path(__file__).resolve().parent / "RESULTS.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
