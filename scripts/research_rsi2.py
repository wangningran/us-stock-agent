"""RSI(2) 深入研究：在基础规则上叠加技术指标过滤。

训练期（挑选指标）：2019-01 ~ 2024-12
测试期（最终检验）：2025-01 ~ 至今

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

# 叠加在 BASE 上的过滤条件
FILTERS = {
    "（基础，无过滤）": lambda d, m: True,
    "布林：跌破下轨": lambda d, m: d.bb_pctb < 0,
    "布林：%B<0.1": lambda d, m: d.bb_pctb < 0.1,
    "MACD 线 > 0": lambda d, m: d.macd > 0,
    "MACD 柱 > 0": lambda d, m: d.macd_hist > 0,
    "MACD 柱 < 0": lambda d, m: d.macd_hist < 0,
    "ADX > 25（强趋势）": lambda d, m: d.adx > 25,
    "ADX < 20（弱趋势）": lambda d, m: d.adx < 20,
    "收盘 > 50日线": lambda d, m: d.close > d.sma50,
    "EMA6 > EMA12": lambda d, m: d.ema6 > d.ema12,
    "随机指标 %K < 20": lambda d, m: d.stoch_k < 20,
    "放量 > 1.5倍": lambda d, m: d.vol_ratio > 1.5,
    "缩量 < 1倍": lambda d, m: d.vol_ratio < 1.0,
    "非跳空下跌(缺口>-3%)": lambda d, m: d.gap > -0.03,
    "单日跌幅 < 5%": lambda d, m: d.ret1 > -0.05,
    "波动率 ATR% < 3%": lambda d, m: d.atr_pct < 0.03,
    "距10日高点回撤 > 5%": lambda d, m: d.dd10 < -0.05,
    "连跌 ≥ 3 天": lambda d, m: d.down_days >= 3,
    "大盘 SPY > 200日线": lambda d, m: m.spy_above200,
    "大盘 SPY > 50日线": lambda d, m: m.spy_above50,
    "大盘也超卖 SPY RSI2<30": lambda d, m: m.spy_rsi2 < 30,
    "VIX < 25": lambda d, m: m.vix < 25,
    "VIX > 20（恐慌时）": lambda d, m: m.vix > 20,
}

HDR = (f"{'过滤条件':<26}{'段':<4}{'笔数':>5}{'胜率':>7}{'均盈':>7}{'均亏':>7}{'每笔':>7}"
       f"{'利润因子':>7}{'年化':>8}{'回撤':>8}{'夏普':>6}{'最差':>8}")


def fmt(name, seg, m):
    if not m.get("trades"):
        return f"{name:<26}{seg:<4}  无交易"
    return (f"{name:<26}{seg:<4}{m['trades']:>5}{m['win_rate']:>7.1%}{m['avg_win']:>7.1%}{m['avg_loss']:>7.1%}"
            f"{m['avg_ret']:>+7.2%}{m['pf']:>7.2f}{m['cagr']:>+8.1%}{m['mdd']:>8.1%}{m['sharpe']:>6.2f}{m['worst']:>8.1%}")


def evaluate(base, mkt, bench, members, entry, exit_=EXIT, periods=(("训练", TRAIN), ("测试", TEST)), **kw):
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
    return (f"{'SPY 买入持有':<26}{seg:<4}{'':>5}{'':>7}{'':>7}{'':>7}{'':>7}{'':>7}"
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
    print("== 单个过滤条件 ==")
    print(HDR)
    for name, f in FILTERS.items():
        r = evaluate(base, mkt, bench, members, lambda d, m, f=f: BASE(d, m) & f(d, m))
        for seg in ("训练", "测试"):
            print(fmt(name, seg, r[seg]), flush=True)
    for seg, (s, e) in (("训练", TRAIN), ("测试", TEST)):
        print(spy_row(bench, seg, s, e))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


# ---- 第二步：组合（只用训练期表现好的三个条件）----
STRESS = lambda d, m: (m.vix > 20) | (m.spy_rsi2 < 30)  # noqa: E731
COMBOS = {
    "基础": lambda d, m: BASE(d, m),
    "VIX>20 + MACD柱>0": lambda d, m: BASE(d, m) & (m.vix > 20) & (d.macd_hist > 0),
    "SPY超卖 + MACD柱>0": lambda d, m: BASE(d, m) & (m.spy_rsi2 < 30) & (d.macd_hist > 0),
    "市场压力(VIX>20或SPY超卖)": lambda d, m: BASE(d, m) & STRESS(d, m),
    "市场压力 + MACD柱>0": lambda d, m: BASE(d, m) & STRESS(d, m) & (d.macd_hist > 0),
}
EXITS = {
    "收盘>5日线": EXIT,
    "RSI2>70": lambda d, m: d.rsi2 > 70,
}


def combos():
    base, mkt, bench, members = load_all()
    print("== 组合 × 出场 × 仓位 ==")
    print(HDR)
    for cname, entry in COMBOS.items():
        for ename, ex in EXITS.items():
            for npos in (10, 5):
                for stop in (None, 0.08):
                    r = evaluate(base, mkt, bench, members, entry, ex, max_positions=npos, stop_pct=stop)
                    tag = f"{cname}|{ename}|{npos}只|{'止损8%' if stop else '无止损'}"
                    for seg in ("训练", "测试"):
                        print(fmt(tag, seg, r[seg]).replace(f"{tag:<26}", tag + "  "), flush=True)
    for seg, (s, e) in (("训练", TRAIN), ("测试", TEST)):
        print(spy_row(bench, seg, s, e))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "combos":
    combos()
