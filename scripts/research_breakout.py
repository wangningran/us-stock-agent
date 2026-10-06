"""Breakout strategy research: squeeze / base breakout on the point-in-time S&P 500.

  python scripts/research_breakout.py            # grid on train (2019-2024) and test (2025-)
  python scripts/research_breakout.py spcx       # check the rules on SPCX
"""
import logging
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent.breakout import bo_features, bo_summary, run_bo_backtest  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.data import get_prices, load_universe  # noqa: E402
from agent.live_rsi2 import entry_rule, exit_rule, rank_rule  # noqa: E402
from agent.meanrev import (add_dollar_volume_rank, build_custom, ext_features, market_context,  # noqa: E402
                           run_mr_backtest)

TRAIN = ("2019-01-01", "2025-01-01")
TEST = ("2025-01-01", "2027-01-01")
ENTRIES = {
    "donchian_v1.5": lambda d: d.breakout_raw & (d.vol_ratio >= 1.5),
    "squeeze_v1.2": lambda d: d.breakout_raw & d.squeeze_recent & (d.vol_ratio >= 1.2),
    "setup_v1.2": lambda d: d.breakout_raw & d.setup_recent & (d.vol_ratio >= 1.2),
}


def load():
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = load_config()
    cfg["universe_mode"] = "sp500_pit"
    prices, _, bench, members = load_universe(cfg)
    vix = get_prices("^VIX", cfg["data"]["start"], cfg["data"]["cache_dir"])
    feats = {t: bo_features(p, bench.close) for t, p in prices.items()}
    add_dollar_volume_rank(feats, bench.index, members)
    return prices, feats, bench, members, vix


def with_entry(feats, members, bench, rule, top_n, mkt_filter):
    spy_ok = bench.close > bench.close.rolling(200).mean()
    out = {}
    for t, d in feats.items():
        m = members[t].reindex(d.index).fillna(False).astype(bool) if t in members else False
        e = rule(d) & m & (d.dv_rank <= top_n)
        if mkt_filter:
            e &= spy_ok.reindex(d.index).fillna(False)
        d = d.copy()
        d["entry"] = e.fillna(False).astype(bool)
        out[t] = d
    return out


def period_days(bench, s, e):
    i = bench.index
    return i[(i >= s) & (i < e)]


def spy_stats(bench, s, e):
    c = bench.close[(bench.index >= s) & (bench.index < e)]
    yrs = (c.index[-1] - c.index[0]).days / 365.25
    return (c.iloc[-1] / c.iloc[0]) ** (1 / yrs) - 1, (c / c.cummax() - 1).min()


HDR = (f"{'config':<44}{'seg':<6}{'trades':>6}{'win':>6}{'avgW':>7}{'avgL':>7}{'payoff':>7}{'PF':>6}"
       f"{'days':>5}{'CAGR':>7}{'MDD':>7}{'Sh':>5}{'best':>6}{'inMkt':>6}")


def fmt(name, seg, s):
    if not s.get("trades"):
        return f"{name:<44}{seg:<6}{0:>6}"
    return (f"{name:<44}{seg:<6}{s['trades']:>6}{s['win_rate']:>6.0%}{s['avg_win']:>7.1%}{s['avg_loss']:>7.1%}"
            f"{s['payoff']:>7.2f}{s['pf']:>6.2f}{s['days']:>5.0f}{s['cagr']:>7.1%}{s['mdd']:>7.1%}{s['sharpe']:>5.2f}"
            f"{s['best']:>6.0%}{s['in_mkt']:>6.0%}")


def grid():
    prices, feats, bench, members, vix = load()
    rows = []
    print(HDR)
    for (ename, rule), stop, ex, top_n, mf in product(ENTRIES.items(), ("day_low", "base_mid"), (10, 20),
                                                       (150, 300), (True,)):
        fe = with_entry(feats, members, bench, rule, top_n, mf)
        name = f"{ename}|{stop}|sma{ex}|top{top_n}{'|spy200' if mf else ''}"
        for seg, (s, e) in (("train", TRAIN), ("test", TEST)):
            tr, eq = run_bo_backtest(fe, period_days(bench, s, e), stop=stop, exit_sma=ex)
            st = bo_summary(tr, eq)
            print(fmt(name, seg, st), flush=True)
            rows.append(dict(config=name, seg=seg, **st))
    for seg, (s, e) in (("train", TRAIN), ("test", TEST)):
        cg, md = spy_stats(bench, s, e)
        print(f"{'SPY buy & hold':<44}{seg:<6}{'':>53}{cg:>7.1%}{md:>7.1%}")
    out = Path("reports/research"); out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "breakout_grid.csv", index=False)


def spcx():
    import yfinance as yf
    raw = yf.download(["SPCX", "SPY"], period="1y", auto_adjust=True, progress=False, group_by="ticker")
    px = raw["SPCX"].rename(columns=str.lower).dropna()
    spy = raw["SPY"]["Close"]
    d = bo_features(px, spy)
    cols = ["close", "bbw", "bbw_pct", "squeeze", "contracting", "higher_lows", "dry", "near_high", "rs63",
            "watch", "base_hi", "vol_ratio", "breakout_raw"]
    print(d[cols].tail(15).round(3).to_string())


if __name__ == "__main__":
    spcx() if len(sys.argv) > 1 and sys.argv[1] == "spcx" else grid()
