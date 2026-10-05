"""Market recap reports (Chinese): regular-session close or after-hours close.

  python scripts/market_recap.py close        # run after 16:00 ET (13:00 PT)
  python scripts/market_recap.py after        # run after 20:00 ET (17:00 PT)
  python scripts/market_recap.py close --no-save
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from agent.config import ROOT  # noqa: E402
from agent.recap import build_afterhours, build_close  # noqa: E402
from agent.universe import current_members, load_sp500_history  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["close", "after"])
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    today = pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    members = current_members(load_sp500_history())
    report = (build_close if args.mode == "close" else build_afterhours)(today, members)
    print(report)
    if not args.no_save:
        out = ROOT / "reports" / "market" / f"{today:%Y-%m-%d}-{args.mode}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report)
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
