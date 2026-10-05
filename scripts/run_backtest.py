"""Backtest entry point for the legacy v1 analyst + momentum strategy.

  python scripts/run_backtest.py                      # real data (needs Yahoo Finance access)
  python scripts/run_backtest.py --synthetic          # synthetic data, offline pipeline check
  python scripts/run_backtest.py --split 2024-01-01   # override the out-of-sample start in config
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backtest import run_backtest  # noqa: E402
from agent.config import ROOT, load_config  # noqa: E402
from agent.metrics import format_summary, summarize  # noqa: E402
from agent.signals import build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--split", help="out-of-sample start date; reports in-sample and out-of-sample separately")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        from agent.synthetic import make_universe
        prices, events, bench = make_universe()
        members = None
        print("⚠️  synthetic data - pipeline check only, the numbers mean nothing\n")
    else:
        from agent.data import load_universe
        prices, events, bench, members = load_universe(cfg)

    feats = build_features(prices, events, cfg, members)
    bt = cfg.get("backtest", {})
    start = args.start or (None if args.synthetic else bt.get("start"))
    split = args.split or (None if args.synthetic else bt.get("split"))
    periods = [("all", start, args.end)]
    if split:
        periods = [("in_sample", start, split), ("out_of_sample", split, args.end)]

    for name, s, e in periods:
        trades, equity = run_backtest(feats, bench, cfg, start=s, end=e)
        if trades.empty:
            print(f"===== {name}: no trades =====\n")
            continue
        print(f"===== {name}  {equity.index[0]:%Y-%m-%d} ~ {equity.index[-1]:%Y-%m-%d} =====")
        print(format_summary(summarize(trades, equity, bench)))
        print()
        out = ROOT / "reports" / f"trades_{name}.csv"
        trades.to_csv(out, index=False)
        print(f"Trades: {out}\n")


if __name__ == "__main__":
    main()
