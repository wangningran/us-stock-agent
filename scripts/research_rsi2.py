"""RSI(2) research: layer technical filters on top of the base rule.

Train period (filter selection): 2019-01 .. 2024-12
Test period (final check):       2025-01 .. today

  python scripts/research_rsi2.py
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from agent.config import load_config  # noqa: E402
from agent.data import get_prices, load_universe  # noqa: E402
from agent.meanrev import build_custom, ext_features, market_context, mr_summary, run_mr_backtest  # noqa: E402

TRAIN = ("2019-01-01", "2025-01-01")
TEST = ("2025-01-01", None)

BASE = lambda d, m: (d.close > d.sma200) & (d.rsi2 < 10)  # noqa: E731
EXIT = lambda d, m: d.close > d.sma5  # noqa: E731
RANK = lambda d, m: d.rsi2  # noqa: E731

# Filters layered on top of BASE
FILTERS = {
    "(base, no filter)": lambda d, m: True,
    "Bollinger: below lower band": lambda d, m: d.bb_pctb < 0,
    "Bollinger: %B < 0.1": lambda d, m: d.bb_pctb < 0.1,
    "MACD line > 0": lambda d, m: d.macd > 0,
    "MACD hist > 0": lambda d, m: d.macd_hist > 0,
    "MACD hist < 0": lambda d, m: d.macd_hist < 0,
    "ADX > 25 (strong trend)": lambda d, m: d.adx > 25,
    "ADX < 20 (weak trend)": lambda d, m: d.adx < 20,
    "close > 50-day SMA": lambda d, m: d.close > d.sma50,
    "EMA6 > EMA12": lambda d, m: d.ema6 > d.ema12,
    "stochastic %K < 20": lambda d, m: d.stoch_k < 20,
    "volume > 1.5x avg": lambda d, m: d.vol_ratio > 1.5,
    "volume < 1x avg": lambda d, m: d.vol_ratio < 1.0,
    "no gap down (gap > -3%)": lambda d, m: d.gap > -0.03,
    "1-day drop < 5%": lambda d, m: d.ret1 > -0.05,
    "ATR% < 3%": lambda d, m: d.atr_pct < 0.03,
    "> 5% below 10-day high": lambda d, m: d.dd10 < -0.05,
    ">= 3 down days": lambda d, m: d.down_days >= 3,
    "SPY > 200-day SMA": lambda d, m: m.spy_above200,
    "SPY > 50-day SMA": lambda d, m: m.spy_above50,
    "SPY also oversold (RSI2 < 30)": lambda d, m: m.spy_rsi2 < 30,
    "VIX < 25": lambda d, m: m.vix < 25,
    "VIX > 20 (fear)": lambda d, m: m.vix > 20,
}

HDR = (f"{'filter':<34}{'seg':<6}{'trades':>7}{'win':>7}{'avgW':>7}{'avgL':>7}{'avg':>7}"
       f"{'PF':>7}{'CAGR':>8}{'MDD':>8}{'Sharpe':>7}{'worst':>8}")


def fmt(name, seg, m):
    if not m.get("trades"):
        return f"{name:<34}{seg:<6}  no trades"
    return (f"{name:<34}{seg:<6}{m['trades']:>7}{m['win_rate']:>7.1%}{m['avg_win']:>7.1%}{m['avg_loss']:>7.1%}"
            f"{m['avg_ret']:>+7.2%}{m['pf']:>7.2f}{m['cagr']:>+8.1%}{m['mdd']:>8.1%}{m['sharpe']:>7.2f}{m['worst']:>8.1%}")


def evaluate(base, mkt, bench, members, entry, exit_=EXIT, periods=(("train", TRAIN), ("test", TEST)), **kw):
    feats = build_custom(base, mkt, entry, exit_, RANK, members)
    res = {}
    for seg, (s, e) in periods:
        tr, eq = run_mr_backtest(feats, bench, mode="close", start=s, end=e, **kw)
        res[seg] = mr_summary(tr, eq)
    return res


def spy_row(bench, seg, s, e):
    x = bench["close"]
    x = x[(x.index >= s) & ((x.index < e) if e else True)]
    yrs = (x.index[-1] - x.index[0]).days / 365.25
    return (f"{'SPY buy & hold':<34}{seg:<6}{'':>7}{'':>7}{'':>7}{'':>7}{'':>7}{'':>7}"
            f"{(x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1:>+8.1%}{(x / x.cummax() - 1).min():>8.1%}")


def load_all():
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = load_config()
    cfg["universe_mode"] = "sp500_pit"
    prices, _, bench, members = load_universe(cfg)
    vix = get_prices("^VIX", cfg["data"]["start"], cfg["data"]["cache_dir"])
    base = {t: ext_features(p) for t, p in prices.items()}
    return base, market_context(bench, vix), bench, members


def main():
    base, mkt, bench, members = load_all()
    print("== Single filters ==")
    print(HDR)
    for name, f in FILTERS.items():
        r = evaluate(base, mkt, bench, members, lambda d, m, f=f: BASE(d, m) & f(d, m))
        for seg in ("train", "test"):
            print(fmt(name, seg, r[seg]), flush=True)
    for seg, (s, e) in (("train", TRAIN), ("test", TEST)):
        print(spy_row(bench, seg, s, e))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ---- Step 2: combinations (only the three filters that helped in the train period) ----
STRESS = lambda d, m: (m.vix > 20) | (m.spy_rsi2 < 30)  # noqa: E731
COMBOS = {
    "base": lambda d, m: BASE(d, m),
    "VIX>20 + MACD hist>0": lambda d, m: BASE(d, m) & (m.vix > 20) & (d.macd_hist > 0),
    "SPY oversold + MACD hist>0": lambda d, m: BASE(d, m) & (m.spy_rsi2 < 30) & (d.macd_hist > 0),
    "stress (VIX>20 or SPY oversold)": lambda d, m: BASE(d, m) & STRESS(d, m),
    "stress + MACD hist>0": lambda d, m: BASE(d, m) & STRESS(d, m) & (d.macd_hist > 0),
}
EXITS = {
    "close > 5-day SMA": EXIT,
    "RSI2>70": lambda d, m: d.rsi2 > 70,
}


def combos():
    base, mkt, bench, members = load_all()
    print("== Combinations x exits x sizing ==")
    print(HDR)
    for cname, entry in COMBOS.items():
        for ename, ex in EXITS.items():
            for npos in (10, 5):
                for stop in (None, 0.08):
                    r = evaluate(base, mkt, bench, members, entry, ex, max_positions=npos, stop_pct=stop)
                    tag = f"{cname}|{ename}|{npos} pos|{'8% stop' if stop else 'no stop'}"
                    for seg in ("train", "test"):
                        print(fmt(tag, seg, r[seg]).replace(f"{tag:<34}", tag + "  "), flush=True)
    for seg, (s, e) in (("train", TRAIN), ("test", TEST)):
        print(spy_row(bench, seg, s, e))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "combos":
    combos()
