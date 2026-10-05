"""Final candidate: RSI(2) + market stress + MACD histogram > 0. Year-by-year results, 2025/2026 split, SPY blend.

  python scripts/final_rsi2.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.research_rsi2 import BASE, EXIT, RANK, STRESS, load_all  # noqa: E402
from agent.meanrev import build_custom, mr_summary, run_mr_backtest  # noqa: E402

ENTRY = lambda d, m: BASE(d, m) & STRESS(d, m) & (d.macd_hist > 0)  # noqa: E731


def stats(r: pd.Series) -> tuple[float, float, float]:
    eq = (1 + r).cumprod()
    yrs = max((r.index[-1] - r.index[0]).days / 365.25, 1e-9)
    return eq.iloc[-1] - 1, (eq / eq.cummax() - 1).min(), r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan


def main():
    base, mkt, bench, members = load_all()
    feats = build_custom(base, mkt, ENTRY, EXIT, RANK, members)
    spy = bench["close"].pct_change()
    for stop in (None, 0.08):
        tr, eq = run_mr_backtest(feats, bench, mode="close", start="2019-01-01", stop_pct=stop)
        r = eq.pct_change().fillna(0)
        tr["exit_date"] = pd.to_datetime(tr["exit_date"])
        print(f"\n===== {'8% stop' if stop else 'no stop'}: by year =====")
        print(f"{'period':<10}{'trades':>6}{'win':>7}{'avg':>8}{'return':>9}{'MDD':>9}{'Sharpe':>7}"
              f"{'SPY ret':>9}{'SPY MDD':>9}{'SPY Sh':>7}{'50/50':>9}{'days in mkt':>12}")
        periods = [(str(y), f"{y}-01-01", f"{y + 1}-01-01") for y in range(2019, 2027)]
        periods += [("2025+2026", "2025-01-01", "2027-01-01")]
        for name, s, e in periods:
            rr = r[(r.index >= s) & (r.index < e)]
            sr = spy.reindex(rr.index).fillna(0)
            t = tr[(tr.exit_date >= s) & (tr.exit_date < e)]
            tot, mdd, sh = stats(rr)
            stot, smdd, ssh = stats(sr)
            btot, _, _ = stats(0.5 * rr + 0.5 * sr)
            days_in = ((eq.reindex(rr.index) - 0).index.isin(
                pd.DatetimeIndex(np.concatenate([pd.bdate_range(a, b) for a, b in
                                                 zip(pd.to_datetime(tr.entry_date), tr.exit_date)]) if len(tr) else []))).mean()
            win = (t.pnl > 0).mean() if len(t) else np.nan
            print(f"{name:<10}{len(t):>5}{win:>7.1%}{t.ret.mean() if len(t) else np.nan:>+8.2%}{tot:>+9.1%}{mdd:>9.1%}{sh:>6.2f}"
                  f"{stot:>+9.1%}{smdd:>9.1%}{ssh:>7.2f}{btot:>+9.1%}{days_in:>12.0%}")
        m = mr_summary(tr[tr.exit_date >= "2025-01-01"], eq[eq.index >= "2025-01-01"])
        print(f"2025 to date: avg win {m['avg_win']:.1%}, avg loss {m['avg_loss']:.1%}, profit factor {m['pf']:.2f}, "
              f"avg hold {m['days']:.1f} days, worst trade {m['worst']:.1%}")
        worst = tr[tr.exit_date >= "2025-01-01"].nsmallest(5, "ret")[["ticker", "entry_date", "exit_date", "ret", "reason"]]
        print("Worst 5 trades since 2025:\n" + worst.to_string(index=False))


if __name__ == "__main__":
    main()
