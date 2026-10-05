import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.live_rsi2 import default_account, load_state, run_live, save_state  # noqa: E402

EMPTY_POS = pd.DataFrame(columns=["ticker", "entry_date", "entry", "shares", "stop", "provisional"])
EMPTY_TR = pd.DataFrame(columns=["ticker", "entry_date", "entry", "exit_date", "exit", "shares", "pnl", "ret",
                                 "reason", "provisional"])


def _px(closes, start="2025-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1e6})


def _world(last_moves):
    """一只股票长期上涨、近期加速后横盘，最后几天按 last_moves 变化；VIX 高位。"""
    base = list(np.linspace(50, 80, 280))
    rally = [base[-1] * 1.03 ** k for k in range(1, 21)]
    flat = list(rally[-1] * np.cumprod([1.002] * 3))
    stock = _px(base + rally + flat + [flat[-1] * m for m in np.cumprod(last_moves)])
    bench = _px(list(np.linspace(300, 400, len(stock))))
    vix = _px([25.0] * len(stock))
    return {"AAA": stock}, bench, vix


def test_buy_then_reconcile_then_sell(tmp_path):
    # 连跌三天触发 RSI(2) 超卖 → 买入
    prices, bench, vix = _world([0.97, 0.97])
    acct = default_account(10000, 2000)
    d = bench.index[-1]
    rep, acct, pos, tr = run_live(prices, bench, vix, None, acct, EMPTY_POS, EMPTY_TR, d)
    assert "买入 AAA" in rep and len(pos) == 1 and bool(pos.provisional.iloc[0])
    assert acct["spy"]["done"] == 1                       # 首次运行买第一批指数
    save_state(acct, pos, tr, tmp_path)
    acct, pos, tr = load_state(tmp_path)

    # 次日大涨，收盘站上 5 日线 → 卖出；并用正式收盘价校正买入价
    prices2, bench2, vix2 = _world([0.97, 0.97, 1.06])
    d2 = bench2.index[-1]
    rep2, acct, pos, tr = run_live(prices2, bench2, vix2, None, acct, pos, tr, d2)
    assert "卖出 AAA" in rep2 and len(pos) == 0 and len(tr) == 1
    assert tr.reason.iloc[0] == "signal" and tr.ret.iloc[0] > 0


def test_closed_market_no_action():
    prices, bench, vix = _world([0.97, 0.97])
    acct = default_account(10000, 2000)
    rep, acct2, pos, tr = run_live(prices, bench, vix, None, acct, EMPTY_POS, EMPTY_TR,
                                   bench.index[-1] + pd.Timedelta(days=1))
    assert "休市" in rep and pos.empty and acct2["spy"]["done"] == 0
