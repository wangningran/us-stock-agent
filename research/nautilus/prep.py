"""Prepare inputs for the Nautilus Trader cross-check of the live RSI(2) configuration (run in the project env).

Writes to research/nautilus/out/:
  bars_signals.csv  - daily OHLCV (rounded to 4 dp) plus entry / exit / rank for every ticker with a signal
  ref_trades.csv    - trades from our own backtester (agent/meanrev.py) on the same data, zero slippage
  ref_equity.csv    - its daily equity curve

  python research/nautilus/prep.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from agent.config import load_config  # noqa: E402
from agent.data import get_prices, load_universe  # noqa: E402
from agent.live_rsi2 import entry_rule, exit_rule, rank_rule  # noqa: E402
from agent.meanrev import add_dollar_volume_rank, build_custom, ext_features, market_context, run_mr_backtest  # noqa: E402

OUT = Path(__file__).resolve().parent / "out"
START = "2019-01-01"
EQUITY0 = 5260


def clean(px: pd.DataFrame) -> pd.DataFrame:
    """Round to 4 dp (Nautilus price precision) and make every bar internally consistent."""
    p = px[["open", "high", "low", "close", "volume"]].dropna(subset=["close"]).copy()
    for c in ("open", "high", "low", "close"):
        p[c] = p[c].round(4)
    p["open"] = p["open"].fillna(p["close"])
    p["high"] = p[["open", "high", "close"]].max(axis=1)
    p["low"] = p[["open", "low", "close"]].min(axis=1)
    p["volume"] = p["volume"].fillna(0).round()
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    cfg["universe_mode"] = "sp500_pit"
    prices, _, bench, members = load_universe(cfg)
    vix = get_prices("^VIX", cfg["data"]["start"], cfg["data"]["cache_dir"])
    prices = {t: clean(p) for t, p in prices.items()}
    base = {t: ext_features(p) for t, p in prices.items()}
    add_dollar_volume_rank(base, bench.index, members)
    ent = lambda d, m: entry_rule(d, m) & (d.dv_rank <= 150)  # noqa: E731
    feats = build_custom(base, market_context(bench, vix), ent, exit_rule, rank_rule, members)

    tr, eq = run_mr_backtest(feats, bench, mode="close", max_positions=5, time_stop=10, stop_pct=0.08,
                             slippage_bps=0, commission=0, equity0=EQUITY0, max_new_per_day=3, start=START)
    tr.to_csv(OUT / "ref_trades.csv", index=False)
    eq.rename_axis("date").to_csv(OUT / "ref_equity.csv")

    days = bench.index[bench.index >= START]
    rows = []
    order = list(feats)                     # our backtester breaks rank ties by this order
    for i, t in enumerate(order):
        d = feats[t]
        d = d[d.index >= START]
        if not d["entry"].any():
            continue
        x = d[["open", "high", "low", "close", "volume", "entry", "exit", "rank"]].copy()
        x["ticker"], x["order"] = t, i
        rows.append(x.rename_axis("date").reset_index())
    out = pd.concat(rows)
    out = out[out.date.isin(days)]
    out.to_csv(OUT / "bars_signals.csv", index=False)
    pd.Series(days, name="date").to_csv(OUT / "days.csv", index=False)
    print(f"tickers with signals: {out.ticker.nunique()}, bar rows: {len(out)}, days: {len(days)}")
    print(f"reference: {len(tr)} trades, final equity {eq.iloc[-1]:,.2f}")


if __name__ == "__main__":
    main()
