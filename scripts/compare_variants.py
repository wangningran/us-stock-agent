"""参数对照：同一份数据上比较多组规则，分样本内/外输出。

  python scripts/compare_variants.py
  python scripts/compare_variants.py --split 2024-01-01
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

# 名称 -> {"section.key": value}
VARIANTS = {
    "v1 基准": {},
    "去掉分析师信号(纯动量)": {"signals.min_analyst_score": -99},
    "去掉大盘过滤": {"signals.market_filter": False},
    "高胜率 止盈1/止损3 ATR": {"trade.target_atr_mult": 1.0, "trade.stop_atr_mult": 3.0, "trade.breakeven_at_r": 0},
    "高胜率 止盈0.5/止损4 ATR": {"trade.target_atr_mult": 0.5, "trade.stop_atr_mult": 4.0, "trade.breakeven_at_r": 0},
}


def apply(cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for path, v in overrides.items():
        sec, key = path.split(".")
        cfg[sec][key] = v
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--split", default="2024-01-01")
    args = ap.parse_args()

    base = load_config(args.config)
    prices, events, bench = load_universe(base)
    print(f"{'规则':<26}{'段':<5}{'笔数':>5}{'胜率':>8}{'盈亏比':>7}{'期望R':>8}{'利润因子':>8}"
          f"{'年化':>8}{'回撤':>8}{'夏普':>6}")
    for name, ov in VARIANTS.items():
        cfg = apply(base, ov)
        feats = build_features(prices, events, cfg)
        for seg, s, e in [("内", None, args.split), ("外", args.split, None)]:
            trades, equity = run_backtest(feats, bench, cfg, start=s, end=e)
            m = summarize(trades, equity, bench)
            if not m["trades"]:
                print(f"{name:<26}{seg:<5}  无交易")
                continue
            print(f"{name:<26}{seg:<5}{m['trades']:>5}{m['win_rate']:>8.1%}{m['payoff_ratio']:>7.2f}"
                  f"{m['expectancy_r']:>+8.3f}{m['profit_factor']:>8.2f}{m['cagr']:>+8.1%}"
                  f"{m['max_drawdown']:>8.1%}{m['sharpe']:>6.2f}")


if __name__ == "__main__":
    main()
