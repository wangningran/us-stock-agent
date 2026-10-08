"""Look-ahead audit on real data: scramble every bar after a cutoff and check that signals and backtest
results up to the cutoff are unchanged (live RSI(2) configuration and the breakout research rule).

  python scripts/audit_lookahead.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from agent.breakout import bo_features, run_bo_backtest  # noqa: E402
from agent.live_rsi2 import entry_rule, exit_rule, rank_rule  # noqa: E402
from agent.meanrev import add_dollar_volume_rank, build_custom, ext_features, run_mr_backtest  # noqa: E402
from agent.meanrev import market_context  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.data import get_prices, load_universe  # noqa: E402

CUTOFFS = ["2020-03-16", "2022-06-16", "2024-08-05", "2025-04-08"]   # stress days, when signals cluster


def poison(df, cut, seed):
    out = df.astype(float)
    rng = np.random.default_rng(seed)
    after = out.index > cut
    k = int(after.sum())
    if k:
        s = rng.uniform(0.3, 3.0, k)
        for c in ("open", "high", "low", "close"):
            out.loc[after, c] = out.loc[after, c].values * s
        out.loc[after, "volume"] = out.loc[after, "volume"].values * rng.uniform(0.1, 10, k)
    return out


def rsi2_feats(prices, bench, vix, members):
    base = {t: ext_features(p) for t, p in prices.items()}
    add_dollar_volume_rank(base, bench.index, members)
    ent = lambda d, m: entry_rule(d, m) & (d.dv_rank <= 150)  # noqa: E731
    return build_custom(base, market_context(bench, vix), ent, exit_rule, rank_rule, members)


def bo_feats(prices, bench, members):
    feats = {t: bo_features(p, bench.close) for t, p in prices.items()}
    add_dollar_volume_rank(feats, bench.index, members)
    spy_ok = bench.close > bench.close.rolling(200).mean()
    for t, d in feats.items():
        m = members[t].reindex(d.index).fillna(False).astype(bool)
        d["entry"] = (d.breakout_raw & d.setup_recent & (d.vol_ratio >= 1.2) & m & (d.dv_rank <= 150)
                      & spy_ok.reindex(d.index).fillna(False)).fillna(False).astype(bool)
    return feats


def compare(tr_a, eq_a, tr_b, eq_b, cut):
    ok = eq_a[eq_a.index <= cut].equals(eq_b[eq_b.index <= cut])
    closed = lambda tr: tr[pd.to_datetime(tr.exit_date) <= cut].sort_values(["entry_date", "ticker"]).reset_index(drop=True)  # noqa: E731
    opened = lambda tr: tr[pd.to_datetime(tr.entry_date) <= cut].sort_values(["entry_date", "ticker"])[  # noqa: E731
        ["ticker", "entry_date", "entry"]].reset_index(drop=True)
    ok &= closed(tr_a).equals(closed(tr_b)) and opened(tr_a).equals(opened(tr_b))
    return ok, len(closed(tr_a))


def main():
    cfg = load_config()
    cfg["universe_mode"] = "sp500_pit"
    prices, _, bench, members = load_universe(cfg)
    vix = get_prices("^VIX", cfg["data"]["start"], cfg["data"]["cache_dir"])
    kw = dict(mode="close", max_positions=5, time_stop=10, stop_pct=0.08, slippage_bps=5, commission=0,
              equity0=5260, max_new_per_day=3, start="2019-01-01")
    fa = rsi2_feats(prices, bench, vix, members)
    tr_a, eq_a = run_mr_backtest(fa, bench, **kw)
    ba = bo_feats(prices, bench, members)
    days = bench.index[bench.index >= "2019-01-01"]
    btr_a, beq_a = run_bo_backtest(ba, days, stop="day_low", exit_sma=20)
    all_ok = True
    for cut in map(pd.Timestamp, CUTOFFS):
        pp = {t: poison(p, cut, i) for i, (t, p) in enumerate(prices.items())}
        pb, pv = poison(bench, cut, 991), poison(vix, cut, 992)
        fb = rsi2_feats(pp, pb, pv, members)
        sig_ok = all(fa[t]["entry"][:cut].equals(fb[t]["entry"][:cut]) and fa[t]["exit"][:cut].equals(fb[t]["exit"][:cut])
                     for t in fa)
        n_sig = sum(int(fa[t]["entry"][:cut].sum()) for t in fa)
        tr_b, eq_b = run_mr_backtest(fb, pb, **kw)
        bt_ok, n_tr = compare(tr_a, eq_a, tr_b, eq_b, cut)
        bb = bo_feats(pp, pb, members)
        bsig_ok = all(ba[t]["entry"][:cut].equals(bb[t]["entry"][:cut]) for t in ba)
        btr_b, beq_b = run_bo_backtest(bb, days, stop="day_low", exit_sma=20)
        bbt_ok, bn_tr = compare(btr_a, beq_a, btr_b, beq_b, cut)
        print(f"cutoff {cut.date()}: RSI(2) signals {'OK' if sig_ok else 'CHANGED'} ({n_sig} entries), "
              f"backtest {'OK' if bt_ok else 'CHANGED'} ({n_tr} closed trades) | breakout signals "
              f"{'OK' if bsig_ok else 'CHANGED'}, backtest {'OK' if bbt_ok else 'CHANGED'} ({bn_tr} closed trades)",
              flush=True)
        all_ok &= sig_ok and bt_ok and bsig_ok and bbt_ok
    print("RESULT:", "no look-ahead found" if all_ok else "LOOK-AHEAD DETECTED")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
