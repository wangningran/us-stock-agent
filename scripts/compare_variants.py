"""参数对照：比较不同股票池 / 打分 / 出场规则，分样本内、样本外输出。

  python scripts/compare_variants.py
  python scripts/compare_variants.py --only 历史成分股
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
# 名称 -> {"key" 或 "section.key": value}
VARIANTS = {
    "手挑46只 v1打分（旧版）": {"universe_mode": "list", "signals.scoring": "v1"},
    "今日成分股 v2打分": {"universe_mode": "sp500_current"},
    "历史成分股 纯动量": {**PIT, "signals.min_analyst_score": -99},
    "历史成分股 v1打分": {**PIT, "signals.scoring": "v1"},
    "历史成分股 v2打分": {**PIT},
    "历史成分股 v2 门槛3": {**PIT, "signals.min_analyst_score": 3},
    "历史成分股 v2 高胜率1/3": {**PIT, "trade.target_atr_mult": 1.0, "trade.stop_atr_mult": 3.0,
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
    ap.add_argument("--only", help="只跑名称包含该字符串的规则")
    args = ap.parse_args()

    base = load_config(args.config)
    start, split = base["backtest"]["start"], base["backtest"]["split"]
    data_cache: dict[str, tuple] = {}
    print(f"{'规则':<24}{'段':<3}{'笔数':>5}{'胜率':>7}{'盈亏比':>7}{'期望R':>8}{'利润因子':>7}"
          f"{'年化':>8}{'回撤':>8}{'夏普':>6}")
    for name, ov in VARIANTS.items():
        if args.only and args.only not in name:
            continue
        cfg = apply(base, ov)
        mode = cfg.get("universe_mode", "list")
        if mode not in data_cache:
            data_cache[mode] = load_universe(cfg)
        prices, events, bench, members = data_cache[mode]
        feats = build_features(prices, events, cfg, members)
        for seg, s, e in [("内", start, split), ("外", split, None)]:
            trades, equity = run_backtest(feats, bench, cfg, start=s, end=e)
            m = summarize(trades, equity, bench)
            if not m["trades"]:
                print(f"{name:<24}{seg:<3}  无交易")
                continue
            print(f"{name:<24}{seg:<3}{m['trades']:>5}{m['win_rate']:>7.1%}{m['payoff_ratio']:>7.2f}"
                  f"{m['expectancy_r']:>+8.3f}{m['profit_factor']:>7.2f}{m['cagr']:>+8.1%}"
                  f"{m['max_drawdown']:>8.1%}{m['sharpe']:>6.2f}", flush=True)
    b = data_cache[next(iter(data_cache))][2]["close"]
    for seg, s, e in [("内", start, split), ("外", split, None)]:
        x = b[(b.index >= s) & ((b.index < e) if e else True)]
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        print(f"{'SPY 买入持有':<24}{seg:<3}{'':>5}{'':>7}{'':>7}{'':>8}{'':>7}"
              f"{(x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1:>+8.1%}{(x / x.cummax() - 1).min():>8.1%}")


if __name__ == "__main__":
    main()
