"""每日盘前报告。

  python scripts/daily_report.py --equity 50000
  python scripts/daily_report.py --synthetic       # 离线预览格式
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.config import ROOT, load_config  # noqa: E402
from agent.report import build_report, load_positions  # noqa: E402
from agent.signals import build_features  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--equity", type=float, help="账户权益，默认取 config 的 initial_equity")
    ap.add_argument("--positions", default=str(ROOT / "positions.csv"))
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.synthetic:
        from agent.synthetic import make_universe
        prices, events, bench = make_universe()
        members = None
    else:
        from agent.data import load_universe
        prices, events, bench, members = load_universe(cfg)

    feats = build_features(prices, events, cfg, members)
    report = build_report(feats, events, bench, cfg, load_positions(args.positions),
                          equity=args.equity or cfg["risk"]["initial_equity"],
                          check_earnings=not args.synthetic)
    out = ROOT / "reports" / f"{bench.index[-1]:%Y-%m-%d}.md"
    out.write_text(report)
    print(report)
    print(f"\n已保存：{out}")


if __name__ == "__main__":
    main()
