import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ledger  # noqa: E402
from agent.live_rsi2 import POS_COLS, TRADE_COLS, default_account, load_state, run_live, save_state  # noqa: E402
from tests.test_live import _world  # noqa: E402


def _empty():
    acct = default_account(10000, 0)
    pos = pd.DataFrame(columns=POS_COLS)
    tr = pd.DataFrame(columns=TRADE_COLS)
    tr["confirmed"] = tr["confirmed"].astype(bool)
    return acct, pos, tr


def test_confirm_buy_adjusts_cash_and_price():
    acct, pos, tr = _empty()
    pos = pd.DataFrame([{"ticker": "MA", "entry_date": pd.Timestamp("2026-10-05"), "entry": 500.0, "shares": 2,
                         "stop": 460.0, "provisional": True, "confirmed": False}])
    acct["cash"] -= 1000
    acct, pos = ledger.confirm_buy(acct, pos, "MA", 1, 505.0, "2026-10-05")
    assert pos.iloc[0].shares == 1 and pos.iloc[0].entry == 505.0 and bool(pos.iloc[0].confirmed)
    assert pos.iloc[0].stop == round(505 * 0.92, 2)
    assert acct["cash"] == pytest.approx(10000 - 505)


def test_cancel_buy_returns_cash():
    acct, pos, tr = _empty()
    acct, pos = ledger.confirm_buy(acct, pos, "AAPL", 3, 200.0)
    acct, pos = ledger.cancel_buy(acct, pos, "AAPL")
    assert pos.empty and acct["cash"] == pytest.approx(10000)


def test_partial_sell_and_cancel_sell():
    acct, pos, tr = _empty()
    acct, pos = ledger.confirm_buy(acct, pos, "NVDA", 10, 100.0)
    acct, pos, tr = ledger.confirm_sell(acct, pos, tr, "NVDA", 4, 110.0)
    assert pos.iloc[0].shares == 6 and len(tr) == 1 and tr.iloc[0].pnl == pytest.approx(40)
    assert acct["cash"] == pytest.approx(10000 - 1000 + 440)
    # 报告记下一笔未确认卖出，用户说没卖 → 放回持仓
    tr = pd.concat([tr, pd.DataFrame([{"ticker": "NVDA", "entry_date": pd.Timestamp("2026-10-01"), "entry": 100.0,
                                       "exit_date": pd.Timestamp("2026-10-06"), "exit": 120.0, "shares": 6, "pnl": 120,
                                       "ret": 0.2, "reason": "signal", "provisional": True, "confirmed": False}])],
                   ignore_index=True)
    pos = pos.iloc[0:0]
    acct["cash"] += 720
    acct, pos, tr = ledger.cancel_sell(acct, pos, tr, "NVDA")
    assert pos.iloc[0].shares == 6 and len(tr) == 1 and acct["cash"] == pytest.approx(9440)


def test_confirmed_price_not_overwritten_by_close(tmp_path):
    prices, bench, vix = _world([0.97, 0.97])
    acct, pos, tr = _empty()
    d = bench.index[-1]
    _, acct, pos, tr = run_live(prices, bench, vix, None, acct, pos, tr, d)
    assert len(pos) == 1
    acct, pos = ledger.confirm_buy(acct, pos, "AAA", int(pos.iloc[0].shares), 123.45, d)
    save_state(acct, pos, tr, tmp_path)
    acct, pos, tr = load_state(tmp_path)
    prices2, bench2, vix2 = _world([0.97, 0.97, 0.99])
    _, acct, pos, tr = run_live(prices2, bench2, vix2, None, acct, pos, tr, bench2.index[-1])
    held = pos[pos.ticker == "AAA"]
    if len(held):
        assert held.iloc[0].entry == 123.45
    else:
        assert tr[tr.ticker == "AAA"].iloc[-1].entry == 123.45
