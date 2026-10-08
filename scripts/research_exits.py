"""Exit-rule research for the live RSI(2) entry (entries unchanged, 8% stop, live sizing).

Train 2019-2024, test 2025-; US$5,260, 5 positions, <= 3 new per day, 5 bp slippage, no commission, no IBIT.

  python scripts/research_exits.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from agent.live_rsi2 import entry_rule, rank_rule  # noqa: E402
from agent.meanrev import add_dollar_volume_rank, build_custom, mr_summary, run_mr_backtest  # noqa: E402
from scripts.research_rsi2 import load_all  # noqa: E402

TRAIN = ("2019-01-01", "2025-01-01")
TEST = ("2025-01-01", "2027-01-01")
NEVER = lambda d, m: d.close < 0  # noqa: E731
sma = lambda d, n: d.close.rolling(n).mean()  # noqa: E731

VARIANTS = {
    "A  close > SMA5 (live)":            (lambda d, m: d.close > d.sma5, {}),
    "B  close > SMA5, hold >= 3 days":   (lambda d, m: d.close > d.sma5, dict(min_hold=3)),
    "C  close > SMA10":                  (lambda d, m: d.close > sma(d, 10), {}),
    "D  close > SMA20":                  (lambda d, m: d.close > sma(d, 20), dict(time_stop=20)),
    "E  RSI(2) > 70":                    (lambda d, m: d.rsi2 > 70, {}),
    "F  RSI(2) > 90":                    (lambda d, m: d.rsi2 > 90, {}),
    "G  close > prior high":             (lambda d, m: d.close > d.high.shift(1), {}),
    "H  ride: back below SMA5 (10d)":    (lambda d, m: (d.close < d.sma5) & (d.close.shift(1) >= d.sma5.shift(1)), {}),
    "I  ride: back below SMA5 (20d)":    (lambda d, m: (d.close < d.sma5) & (d.close.shift(1) >= d.sma5.shift(1)),
                                          dict(time_stop=20)),
    "J  target +3%":                     (NEVER, dict(target_pct=0.03)),
    "K  target +5%":                     (NEVER, dict(target_pct=0.05)),
    "L  target +8% (20d)":               (NEVER, dict(target_pct=0.08, time_stop=20)),
    "M  SMA5 or +5% target":             (lambda d, m: d.close > d.sma5, dict(target_pct=0.05)),
    "N  close > SMA5, 20d time stop":    (lambda d, m: d.close > d.sma5, dict(time_stop=20)),
}

HDR = (f"{'exit rule':<36}{'seg':<6}{'trades':>6}{'win':>6}{'avg':>7}{'avgW':>7}{'avgL':>7}{'days':>5}"
       f"{'CAGR':>7}{'MDD':>7}{'Sh':>6}")


def fmt(name, seg, s):
    return (f"{name:<36}{seg:<6}{s['trades']:>6}{s['win_rate']:>6.0%}{s['avg_ret']:>+7.2%}{s['avg_win']:>+7.1%}"
            f"{s['avg_loss']:>+7.1%}{s['days']:>5.1f}{s['cagr']:>+7.1%}{s['mdd']:>7.1%}{s['sharpe']:>6.2f}")


def main():
    base, mkt, bench, members = load_all()
    add_dollar_volume_rank(base, bench.index, members)
    ent = lambda d, m: entry_rule(d, m) & (d.dv_rank <= 150)  # noqa: E731
    rows = []
    print(HDR)
    for name, (ex, extra) in VARIANTS.items():
        feats = build_custom(base, mkt, ent, ex, rank_rule, members)
        for seg, (s, e) in (("train", TRAIN), ("test", TEST)):
            kw = dict(mode="close", max_positions=5, time_stop=10, stop_pct=0.08, slippage_bps=5, commission=0,
                      equity0=5260, max_new_per_day=3, start=s, end=e)
            kw.update(extra)
            tr, eq = run_mr_backtest(feats, bench, **kw)
            st = mr_summary(tr, eq)
            print(fmt(name, seg, st), flush=True)
            rows.append(dict(rule=name, seg=seg, **st))
    out = Path("reports/research")
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "exit_rules.csv", index=False)


if __name__ == "__main__":
    main()
