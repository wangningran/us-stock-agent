"""短线策略对照：均值回归（RSI2 / 累积RSI2 / IBS）与短均线（EMA6/12）。

  python scripts/compare_meanrev.py
  python scripts/compare_meanrev.py --universe sp500_current   # 对比幸存者偏差
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import load_config  # noqa: E402
from agent.data import load_universe  # noqa: E402
from agent.meanrev import RULES, build_mr, mr_summary, run_mr_backtest  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="sp500_pit")
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--modes", default="close,open")
    ap.add_argument("--stop", type=float, default=None, help="硬止损比例，如 0.1")
    args = ap.parse_args()

    cfg = load_config()
    cfg["universe_mode"] = args.universe
    prices, _, bench, members = load_universe(cfg)
    start, split = cfg["backtest"]["start"], cfg["backtest"]["split"]
    hdr = f"{'规则':<20}{'成交':<6}{'段':<3}{'笔数':>6}{'胜率':>7}{'均盈':>7}{'均亏':>7}{'每笔':>7}{'利润因子':>7}{'天':>5}{'年化':>8}{'回撤':>8}{'夏普':>6}{'最差一笔':>9}"
    print(hdr)
    for rule in args.rules.split(","):
        feats = build_mr(prices, rule, members)
        for mode in args.modes.split(","):
            for seg, s, e in [("内", start, split), ("外", split, None)]:
                tr, eq = run_mr_backtest(feats, bench, mode=mode, stop_pct=args.stop, start=s, end=e)
                m = mr_summary(tr, eq)
                if not m["trades"]:
                    print(f"{rule:<20}{mode:<6}{seg:<3}  无交易"); continue
                print(f"{rule:<20}{mode:<6}{seg:<3}{m['trades']:>6}{m['win_rate']:>7.1%}{m['avg_win']:>7.1%}{m['avg_loss']:>7.1%}"
                      f"{m['avg_ret']:>+7.2%}{m['pf']:>7.2f}{m['days']:>5.1f}{m['cagr']:>+8.1%}{m['mdd']:>8.1%}{m['sharpe']:>6.2f}{m['worst']:>9.1%}",
                      flush=True)
    b = bench["close"]
    for seg, s, e in [("内", start, split), ("外", split, None)]:
        x = b[(b.index >= s) & ((b.index < e) if e else True)]
        yrs = (x.index[-1] - x.index[0]).days / 365.25
        print(f"{'SPY 买入持有':<20}{'':<6}{seg:<3}{'':>6}{'':>7}{'':>7}{'':>7}{'':>7}{'':>7}{'':>5}"
              f"{(x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1:>+8.1%}{(x / x.cummax() - 1).min():>8.1%}")


if __name__ == "__main__":
    main()
