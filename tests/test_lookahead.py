"""Look-ahead tests: changing data after a cutoff must not change anything computed up to the cutoff.

Covers features, live RSI(2) signals, dollar-volume rank, the RSI(2) backtester and the breakout research code.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.breakout import bo_features, run_bo_backtest  # noqa: E402
from agent.live_rsi2 import entry_rule, exit_rule, rank_rule  # noqa: E402
from agent.meanrev import (add_dollar_volume_rank, build_custom, ext_features, market_context,  # noqa: E402
                           run_mr_backtest)
from agent.synthetic import make_prices, make_universe  # noqa: E402

N_DAYS = 900
CUT = 650          # index of the cutoff day


def _world(seed=0):
    prices, _, bench = make_universe(n_tickers=12, n_days=N_DAYS, seed=seed)
    # Volatile stocks so RSI(2) dips and breakouts actually occur
    for i, t in enumerate(list(prices)):
        prices[t] = make_prices(N_DAYS, seed + i, drift=0.0012, vol=0.028)
    vix = bench.copy()
    rng = np.random.default_rng(seed + 7)
    v = 21 + rng.normal(0, 3, len(vix))
    for col in ("open", "high", "low", "close"):
        vix[col] = v
    members = pd.DataFrame(True, index=bench.index, columns=list(prices))
    return prices, bench, vix, members


def _poison(df: pd.DataFrame, cut_day, seed=1) -> pd.DataFrame:
    """Replace every bar after cut_day with garbage of a different scale (keeps the index)."""
    out = df.astype(float)
    rng = np.random.default_rng(seed)
    after = out.index > cut_day
    k = int(after.sum())
    scale = rng.uniform(0.3, 3.0, k)
    for col in ("open", "high", "low", "close"):
        if col in out:
            out.loc[after, col] = out.loc[after, col].values * scale
    if "volume" in out:
        out.loc[after, "volume"] = out.loc[after, "volume"].values * rng.uniform(0.1, 10, k)
    return out


def _same(a: pd.Series, b: pd.Series, cut_day, name):
    a, b = a[a.index <= cut_day], b[b.index <= cut_day]
    pd.testing.assert_series_equal(a, b, check_names=False, obj=name)


def _live_feats(prices, bench, vix, members):
    base = {t: ext_features(p) for t, p in prices.items()}
    add_dollar_volume_rank(base, bench.index, members)
    mkt = market_context(bench, vix)
    ent = lambda d, m: entry_rule(d, m) & (d.dv_rank <= 8)  # noqa: E731
    return build_custom(base, mkt, ent, exit_rule, rank_rule, members)


def test_rsi2_features_and_signals_ignore_future():
    prices, bench, vix, members = _world()
    cut_day = bench.index[CUT]
    full = _live_feats(prices, bench, vix, members)
    pois = _live_feats({t: _poison(p, cut_day, i) for i, (t, p) in enumerate(prices.items())},
                       _poison(bench, cut_day, 99), _poison(vix, cut_day, 98), members)
    n_entries = 0
    for t in prices:
        for col in ("sma5", "sma200", "rsi2", "macd_hist", "dv_rank", "entry", "exit", "rank"):
            _same(full[t][col], pois[t][col], cut_day, f"{t}.{col}")
        n_entries += int(full[t]["entry"][:cut_day].sum())
    assert n_entries > 5, "test world should produce RSI(2) entries"


def _check_trades(tr_a, tr_b, cut_day, name):
    """Trades closed by the cutoff are identical; trades opened by the cutoff have identical entries."""
    for col, how in (("exit_date", "closed"), ("entry_date", "opened")):
        a = tr_a[pd.to_datetime(tr_a[col]) <= cut_day].sort_values(["entry_date", "ticker"]).reset_index(drop=True)
        b = tr_b[pd.to_datetime(tr_b[col]) <= cut_day].sort_values(["entry_date", "ticker"]).reset_index(drop=True)
        cols = list(a.columns) if how == "closed" else ["ticker", "entry_date", "entry"]
        pd.testing.assert_frame_equal(a[cols], b[cols], obj=f"{name} trades {how} by {cut_day.date()}")
    return int((pd.to_datetime(tr_a.exit_date) <= cut_day).sum())


def _event_cutoffs(tr, n=12):
    """Cutoff days that land on entries and stop exits (where a peek at tomorrow would change a fill)."""
    days = sorted(set(pd.to_datetime(tr.entry_date)) | set(pd.to_datetime(tr[tr.reason == "stop"].exit_date)))
    days = days[5:-5]
    step = max(1, len(days) // n)
    return days[::step][:n] + list(pd.to_datetime(tr[tr.reason == "stop"].exit_date))[2:6]


def test_rsi2_backtester_ignores_future():
    prices, bench, vix, members = _world()
    kw = dict(mode="close", max_positions=5, time_stop=10, stop_pct=0.08, slippage_bps=5, commission=0,
              equity0=5260, max_new_per_day=3)
    full = _live_feats(prices, bench, vix, members)
    tr_a, eq_a = run_mr_backtest(full, bench, **kw)
    checked = 0
    assert (tr_a.reason == "stop").sum() >= 3, "test world should produce stop exits"
    for cut_day in _event_cutoffs(tr_a):
        pois = _live_feats({t: _poison(p, cut_day, i) for i, (t, p) in enumerate(prices.items())},
                           _poison(bench, cut_day, 99), _poison(vix, cut_day, 98), members)
        tr_b, eq_b = run_mr_backtest(pois, _poison(bench, cut_day, 99), **kw)
        _same(eq_a, eq_b, cut_day, "equity")
        checked = max(checked, _check_trades(tr_a, tr_b, cut_day, "rsi2"))
    assert checked > 3


def _bo_feats(px, spy):
    out = {}
    for t, p in px.items():
        d = bo_features(p, spy.close)
        d["entry"] = (d.breakout_raw & (d.vol_ratio >= 1.2)).fillna(False)
        out[t] = d
    return out


def test_breakout_features_and_backtester_ignore_future():
    prices, bench, _, _ = _world(seed=3)
    days = bench.index[250:]
    full = _bo_feats(prices, bench)
    tr_a, eq_a = run_bo_backtest(full, days, stop="day_low", exit_sma=20)
    assert (tr_a.reason == "stop").sum() >= 3
    checked = 0
    for k, cut_day in enumerate(_event_cutoffs(tr_a)):
        pois = _bo_feats({t: _poison(p, cut_day, i) for i, (t, p) in enumerate(prices.items())},
                         _poison(bench, cut_day, 99))
        if k == 0:
            for t in prices:
                for col in ("bbw_pct", "base_hi", "base_lo", "rs63", "vol_ratio", "watch", "setup_recent", "entry"):
                    _same(full[t][col], pois[t][col], cut_day, f"{t}.{col}")
        tr_b, eq_b = run_bo_backtest(pois, days, stop="day_low", exit_sma=20)
        _same(eq_a, eq_b, cut_day, "breakout equity")
        checked = max(checked, _check_trades(tr_a, tr_b, cut_day, "breakout"))
    assert checked > 3


def test_entry_fill_never_before_signal_and_stop_not_same_day():
    prices, bench, vix, members = _world()
    feats = _live_feats(prices, bench, vix, members)
    tr, _ = run_mr_backtest(feats, bench, mode="close", max_positions=5, stop_pct=0.08, commission=0,
                            equity0=5260, max_new_per_day=3)
    for r in tr.itertuples():
        assert bool(feats[r.ticker].at[r.entry_date, "entry"]), "entry only on a signal day"
        if r.reason == "stop":
            assert r.exit_date > r.entry_date, "stop cannot trigger on the entry bar"
