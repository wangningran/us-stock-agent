"""Live daily report (paper trading): RSI(2) strategy on the model account in state/.

  python scripts/live_report.py --init --strategy-capital 5260 --spy-budget 0   # initialise the model account
  python scripts/live_report.py                    # generate today's report and update state/
  python scripts/live_report.py --dry-run          # print the report without changing state
  python scripts/live_report.py --today 2026-10-02 --dry-run   # replay a past day (data truncated)
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from agent.config import ROOT, load_config  # noqa: E402
from agent.data import get_prices, load_universe  # noqa: E402
from agent.universe import current_members, load_sp500_history  # noqa: E402
from agent.live_rsi2 import POS_COLS, STATE_DIR, TRADE_COLS, default_account, load_state, run_live, save_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--strategy-capital", type=float, default=2630)
    ap.add_argument("--spy-budget", type=float, default=2630)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--today", help="US/Eastern date YYYY-MM-DD, default today")
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    args = ap.parse_args()
    state_dir = Path(args.state_dir)

    if args.init:
        if (state_dir / "account.json").exists():
            sys.exit(f"{state_dir}/account.json already exists; delete it first to reset")
        save_state(default_account(args.strategy_capital, args.spy_budget),
                   pd.DataFrame(columns=POS_COLS), pd.DataFrame(columns=TRADE_COLS), state_dir)
        print(f"Model account initialised: strategy ${args.strategy_capital:,.0f}, index sleeve ${args.spy_budget:,.0f}")
        return

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = load_config()
    cfg["universe_mode"] = "sp500_pit"
    prices, _, bench, members = load_universe(cfg)
    vix = get_prices("^VIX", cfg["data"]["start"], cfg["data"]["cache_dir"])
    today = pd.Timestamp(args.today) if args.today else \
        pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    if args.today:  # replay: truncate data to that day
        cut = lambda df: df[df.index <= today]  # noqa: E731
        prices = {t: cut(p) for t, p in prices.items() if len(cut(p))}
        bench, vix = cut(bench), cut(vix)
        if members is not None:
            members = members[members.index <= today]

    # Refuse to report on a partial universe (e.g. Yahoo rate limiting) rather than give wrong signals
    if members is not None and not args.today:
        current = current_members(load_sp500_history())
        have = sum(t in prices for t in current)
        if have / len(current) < 0.9:
            sys.exit(f"ERROR: only {have}/{len(current)} current index members have price data; aborting")
    acct, pos, trades = load_state(state_dir)
    core = None
    if acct["spy"].get("budget", 0) > 0:   # index sleeve disabled when its budget is zero
        core = get_prices(acct["spy"].get("ticker", "SPY"), cfg["data"]["start"], cfg["data"]["cache_dir"])
        if args.today:
            core = core[core.index <= today]
    extras = {}
    for t in acct.get("extras", []):
        x = get_prices(t, cfg["data"]["start"], cfg["data"]["cache_dir"])
        extras[t] = x[x.index <= today] if args.today else x
    report, acct, pos, trades = run_live(prices, bench, vix, members, acct, pos, trades, today, core, extras)
    print(report)
    if not args.dry_run:
        save_state(acct, pos, trades, state_dir)
        out = ROOT / "reports" / "live" / f"{today:%Y-%m-%d}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
