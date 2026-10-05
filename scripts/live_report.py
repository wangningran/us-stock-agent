"""实盘每日报告（模拟盘）：RSI(2) 策略 + SPY 分批建仓。

  python scripts/live_report.py --init --strategy-capital 2630 --spy-budget 2630   # 初始化模型账户
  python scripts/live_report.py                    # 生成今日报告并更新 state/
  python scripts/live_report.py --dry-run          # 只看报告，不更新状态
  python scripts/live_report.py --today 2026-10-02 --dry-run   # 用历史某天回放（截断数据）
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from agent.config import ROOT, load_config  # noqa: E402
from agent.data import get_prices, load_universe  # noqa: E402
from agent.live_rsi2 import POS_COLS, STATE_DIR, TRADE_COLS, default_account, load_state, run_live, save_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--strategy-capital", type=float, default=2630)
    ap.add_argument("--spy-budget", type=float, default=2630)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--today", help="美东日期 YYYY-MM-DD，默认今天")
    ap.add_argument("--state-dir", default=str(STATE_DIR))
    args = ap.parse_args()
    state_dir = Path(args.state_dir)

    if args.init:
        if (state_dir / "account.json").exists():
            sys.exit(f"{state_dir}/account.json 已存在，如需重置请先手动删除")
        save_state(default_account(args.strategy_capital, args.spy_budget),
                   pd.DataFrame(columns=POS_COLS), pd.DataFrame(columns=TRADE_COLS), state_dir)
        print(f"已初始化模型账户：策略 ${args.strategy_capital:,.0f}，SPY 预算 ${args.spy_budget:,.0f}")
        return

    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    cfg = load_config()
    cfg["universe_mode"] = "sp500_pit"
    prices, _, bench, members = load_universe(cfg)
    vix = get_prices("^VIX", cfg["data"]["start"], cfg["data"]["cache_dir"])
    today = pd.Timestamp(args.today) if args.today else \
        pd.Timestamp.now(tz="America/New_York").normalize().tz_localize(None)
    if args.today:  # 回放：截断到当天
        cut = lambda df: df[df.index <= today]  # noqa: E731
        prices = {t: cut(p) for t, p in prices.items() if len(cut(p))}
        bench, vix = cut(bench), cut(vix)
        if members is not None:
            members = members[members.index <= today]

    acct, pos, trades = load_state(state_dir)
    core_t = acct["spy"].get("ticker", "SPY")
    core = get_prices(core_t, cfg["data"]["start"], cfg["data"]["cache_dir"])
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
        print(f"\n已保存：{out}")


if __name__ == "__main__":
    main()
