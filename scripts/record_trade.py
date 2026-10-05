"""记录实际成交，让后续卖出建议基于真实持仓。

  python scripts/record_trade.py buy  MA 2 525.30            # 确认买入（实际股数、价格）
  python scripts/record_trade.py buy  AAPL 3 228.10 --date 2026-10-05   # 报告外的买入也可以记
  python scripts/record_trade.py nobuy MA                    # 报告推荐了但没买
  python scripts/record_trade.py sell MA 2 540.00            # 确认卖出
  python scripts/record_trade.py nosell MA                   # 报告建议卖出但没卖
  python scripts/record_trade.py show                        # 查看当前持仓
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import ledger  # noqa: E402
from agent.live_rsi2 import STATE_DIR, load_state, save_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["buy", "nobuy", "sell", "nosell", "show"])
    ap.add_argument("ticker", nargs="?")
    ap.add_argument("shares", nargs="?", type=int)
    ap.add_argument("price", nargs="?", type=float)
    ap.add_argument("--date")
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    a = ap.parse_args()
    sd = Path(a.state_dir)
    acct, pos, trades = load_state(sd)
    t = a.ticker.upper() if a.ticker else None
    if a.action in ("buy", "sell") and (a.shares is None or a.price is None):
        sys.exit("buy/sell 需要：代码 股数 价格")
    if a.action == "buy":
        acct, pos = ledger.confirm_buy(acct, pos, t, a.shares, a.price, a.date)
    elif a.action == "nobuy":
        acct, pos = ledger.cancel_buy(acct, pos, t)
    elif a.action == "sell":
        acct, pos, trades = ledger.confirm_sell(acct, pos, trades, t, a.shares, a.price, a.date)
    elif a.action == "nosell":
        acct, pos, trades = ledger.cancel_sell(acct, pos, trades, t)
    if a.action != "show":
        save_state(acct, pos, trades, sd)
    print(pos.to_string(index=False) if len(pos) else "当前无持仓")
    print(f"现金 ${acct['cash']:,.2f}")
    pend = trades[~trades.confirmed.astype(bool)] if len(trades) else trades
    if len(pend):
        print("待确认卖出：\n" + pend[["ticker", "exit_date", "exit", "shares", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
