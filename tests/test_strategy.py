import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.backtest import run_backtest  # noqa: E402
from agent.config import load_config  # noqa: E402
from agent.signals import build_features, daily_analyst_score, event_score  # noqa: E402
from agent.strategy import OrderPlan, Position, check_exit, try_fill  # noqa: E402
from agent.synthetic import make_universe  # noqa: E402

CFG = load_config()


def bar(o, h, l, c):
    return pd.Series({"open": o, "high": h, "low": l, "close": c})


def order(limit=100.0):
    return OrderPlan("X", pd.Timestamp("2024-01-02"), limit, 2.0, 10, 96, 106, 3, 0.1, 50)


def pos(entry=100.0, stop=96.0, target=106.0):
    return Position("X", pd.Timestamp("2024-01-02"), entry, 10, stop, target, entry - stop)


def test_limit_fill_requires_touch():
    assert try_fill(order(100), bar(102, 103, 100.5, 101)) is None
    assert try_fill(order(100), bar(102, 103, 99, 101)) == 100
    assert try_fill(order(100), bar(98, 103, 97, 101)) == 98  # 低开按开盘价成交


def test_same_bar_stop_and_target_is_stop():
    assert check_exit(pos(), bar(100, 107, 95, 101), CFG) == (96, "stop")


def test_gap_through_stop_exits_at_open():
    assert check_exit(pos(), bar(93, 94, 92, 93), CFG) == (93, "stop_gap")


def test_fill_day_ignores_target():
    assert check_exit(pos(), bar(100, 110, 99, 108), CFG, fill_day=True) is None


def test_time_stop():
    p = pos()
    p.days_held = CFG["trade"]["max_hold_days"]
    assert check_exit(p, bar(100, 101, 99, 100.5), CFG) == (100.5, "time")


def test_event_scoring():
    assert event_score({"action": "up", "pt_action": "Raises"}) == 3
    assert event_score({"action": "down", "pt_action": "Lowers"}) == -3
    assert event_score({"action": "init", "to_grade": "Outperform"}) == 1
    assert event_score({"action": "main", "pt_current": 120, "pt_prior": 100}) == 1


def test_weekend_event_maps_to_next_trading_day():
    days = pd.bdate_range("2024-01-01", periods=10)
    ev = pd.DataFrame({"action": ["up"], "pt_action": ["Raises"]},
                      index=[pd.Timestamp("2024-01-06 10:00")])  # 周六
    s = daily_analyst_score(ev, days)
    assert s[pd.Timestamp("2024-01-08")] == 3 and s.sum() == 3


def test_no_lookahead_entries_after_signal():
    prices, events, bench = make_universe(n_tickers=6, n_days=700)
    feats = build_features(prices, events, CFG)
    trades, equity = run_backtest(feats, bench, CFG)
    assert len(trades) > 0
    assert (pd.to_datetime(trades.entry_date) > pd.to_datetime(trades.signal_date)).all()
    assert (pd.to_datetime(trades.exit_date) >= pd.to_datetime(trades.entry_date)).all()
    assert equity.notna().all() and (equity > 0).all()


def test_future_events_do_not_change_past_signals():
    prices, events, bench = make_universe(n_tickers=4, n_days=700)
    cut = bench.index[500]
    full = build_features(prices, events, CFG)
    trunc_px = {t: p[p.index <= cut] for t, p in prices.items()}
    trunc_ev = {t: e[e.index.normalize() <= cut] for t, e in events.items()}
    part = build_features(trunc_px, trunc_ev, CFG)
    for t in prices:
        pd.testing.assert_series_equal(full[t]["candidate"][:cut], part[t]["candidate"], check_names=False)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
