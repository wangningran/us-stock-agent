"""Compare universes / analyst scoring / exit rules for the legacy v1 strategy, in-sample vs out-of-sample.

  python scripts/compare_variants.py
  python scripts/compare_variants.py --only PIT
"""
import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backtest import run_backtest  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.data import load_universe  # noqa: E402
from agent.metrics import summarize  # noqa: E402
from agent.signals import build_features  # noqa: E402

PIT = {"universe_mode": "sp500_pit"}
# name -> {"key" or "section.key": value}
VARIANTS = {
    "hand-picked 46, v1 scoring": {"universe_mode": "list", "signals.scoring": "v1"},
    "current S&P 500, v2": {"universe_mode": "sp500_current"},
    "PIT S&P 500, momentum only": {**PIT, "signals.min_analyst_score": -99},
    "PIT S&P 500, v1 scoring": {**PIT, "signals.scoring": "v1"},
    "PIT S&P 500, v2 scoring": {**PIT},
    "PIT S&P 500, v2 score>=3": {**PIT, "signals.min_analyst_score": 3},
    "PIT S&P 500, v2 TP1/SL3": {**PIT, "trade.target_atr_mult": 1.0, "trade.stop_atr_mult": 3.0,
                          "trade.breakeven_at_r": 0},
}


def apply(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for path, v in overrides.items():
        if "." in path:
            sec, key = path.split(".")
            cfg[sec][key] = v
        else:
            cfg[path] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--only", help="only run variants whose name contains this string")
    args = ap.parse_args()

    base = load_config(args.config)
    start, split = base["backtest"]["start"], base["backtest"]["split"]
    data_cache: dict[str, tuple] = {}
    print(f"{'variant':<30}{'seg':<5}{'trades':>7}{'win':>7}{'payoff':>7}{'E[R]':>8}{'PF':>7}"
          f"{'CAGR':>8}{'MDD':>8}{'Sharpe':>7}")
    for name, ov in VARIANTS.items():
        if args.only and args.only not in name:
            continue
        cfg = apply(base, ov)
        mode = cfg.get("universe_mode", "list")
        if mode not in data_cache:
            data_cache[mode] = load_universe(cfg)
        prices, events, bench, members = data_cache[mode]
        feats = build_features(prices, events, cfg, members)
        for seg, s, e in [("IS", start, split), ("OOS", split, None)]:
            trades, equity = run_backtest(feats, bench, cfg, start=s, end=e)
            m = summarize(trades, equity, bench)
            if not m["trades"]:
                print(f"{name:<30}{seg:<5}  no trades")
                continue
            print(f"{name:<30}{seg:<5}{m['trades']:>7}{m['win_rate']:>7.1%}{m['payoff_ratio']:>7.2f}"
                  f"{m['expectancy_r']:>+8.3f}{m['profit_factor']:>7.2f}{m['cagr']:>+8.1%}"
                  f"{m['max_drawdown']:>8.1%}{m['sharpe']:>7.2f}", flush=True)
    b = data_cache[next(iter(data_cache))][2]["close"]
    for seg, s, e in [("IS", start, split), ("OOS", split, None)]:
        x = b[(b.index >= s) & ((b.index < e) if e else True)]
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        print(f"{'SPY buy & hold':<30}{seg:<5}{'':>7}{'':>7}{'':>7}{'':>8}{'':>7}"
              f"{(x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1:>+8.1%}{(x / x.cummax() - 1).min():>8.1%}")


if __name__ == "__main__":
    main()
