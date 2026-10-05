"""Record actual fills so later exit advice is based on real positions.

  python scripts/record_trade.py buy  MA 2 525.30            # confirm a buy (actual shares, price)
  python scripts/record_trade.py buy  AAPL 3 228.10 --date 2026-10-05   # buys outside the report work too
  python scripts/record_trade.py nobuy MA                    # recommended but not bought
  python scripts/record_trade.py sell MA 2 540.00            # confirm a sell
  python scripts/record_trade.py nosell MA                   # recommended sell not executed
  python scripts/record_trade.py show                        # show current positions
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
        sys.exit("buy/sell need: TICKER SHARES PRICE")
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
    print(pos.to_string(index=False) if len(pos) else "No positions")
    print(f"Cash ${acct['cash']:,.2f}" + ("  (negative: check the recorded fills)" if acct["cash"] < 0 else ""))
    pend = trades[~trades.confirmed.astype(bool)] if len(trades) else trades
    if len(pend):
        print("Pending (unconfirmed) sells:\n" + pend[["ticker", "exit_date", "exit", "shares", "reason"]].to_string(index=False))


if __name__ == "__main__":
    main()
