# us-stock-agent

A research and paper-trading tool for a short-term **RSI(2) mean-reversion** strategy on large US stocks
(plus Bitcoin via the IBIT ETF). Every trading day it generates an order list ~45 minutes before the close,
which is executed by hand at a broker, and it keeps a model account that tracks the user's actual fills.

> Research and paper trading only. Not investment advice. Backtests overstate real results (see Limitations).

## Live strategy

| Step | Rule |
|---|---|
| Universe | Point-in-time S&P 500 members ranked in the **top 150 by 60-day average dollar volume** (a size / popularity proxy with no look-ahead), plus **IBIT** (iShares Bitcoin Trust) |
| Entry (stocks) | Close > 200-day SMA **and** RSI(2) < 10 **and** market stress (VIX > 20 **or** SPY RSI(2) < 30) **and** MACD histogram > 0 |
| Entry (IBIT) | Close > 200-day SMA **and** RSI(2) < 10 (no stress / MACD / dollar-volume filters) |
| Ranking | Lowest RSI(2) first; at most **3 new buys per day** |
| Sizing | Up to **5 positions**, ~20% of equity each, whole shares |
| Execution | Market-on-close (or a limit at the reference price) before 12:50 PT |
| Exit | Close > 5-day SMA (sell at the close), **8% GTC stop**, or a **10-trading-day** time stop |
| Earnings | Buys and holdings with earnings within 10 trading days are flagged (reminder only) |

Model account: 7,500 CAD ≈ US$5,260, started 2026-10-05 (`state/account.json`, kept locally).

### Backtest of the live configuration

Zero commission, 5 bp slippage, US$5,260, whole shares, point-in-time S&P 500, IBIT simulated with scaled BTC-USD
prices before 2024.

| Period | Trades | Win rate | CAGR | Max drawdown | Sharpe | SPY CAGR | SPY max DD |
|---|---|---|---|---|---|---|---|
| 2019-01 – 2024-12 (train) | 561 | 71.3% | +9.5% | −15.1% | 0.99 | +17.1% | −33.7% |
| 2025-01 – 2026-10 (test) | 135 | 70.4% | +12.7% | −4.8% | 1.37 | +18.4% | −18.8% |

The strategy trades only when the market is under stress, so it can sit in cash for days. It does best in
sell-offs and bear markets (e.g. 2022: +3% vs SPY −18%) and lags SPY in strong bull markets.

## Daily workflow

1. A scheduled job runs `scripts/live_report.py` at **12:12 PT** on weekdays and pushes the report (in Chinese)
   by phone notification and email. Market holidays produce a one-line "market closed" message.
2. The user places market-on-close orders before 12:50 PT and an 8% GTC stop after each buy.
3. The user reports actual fills; they are recorded with `scripts/record_trade.py`. Unreported trades are assumed
   filled at the close (shown as ⏳ unconfirmed); reported ones are ✅ confirmed and never overwritten.
4. State (`state/`) and daily trade reports (`reports/live/`) stay on the machine that runs the report and are
   not committed (they contain the user's own fills). Run `python scripts/live_report.py --init` to start a fresh
   model account, then `scripts/record_trade.py` to re-enter open positions.

Signal reference: [`docs/signals.md`](docs/signals.md).

Sample reports: [`docs/sample_report_buy.md`](docs/sample_report_buy.md),
[`docs/sample_report_sell.md`](docs/sample_report_sell.md).

## Market recap reports

`scripts/market_recap.py` produces two Chinese reports each trading day, saved to `reports/market/`:

- `close` (13:08 PT): indices and macro, breadth (advancers / decliners, 52-week highs / lows, % above 200-day),
  sectors with their largest names and best / worst members, S&P 500 movers of 4% or more, a fixed list of heavily
  traded non-S&P stocks (`HOT_EXTRA`: SpaceX, TSMC, ASML, ...), the RSI(2) watch-list, the breakout watch-list
  and breakout signals (information only, not traded; see research item 6)
  and next-day earnings.
- `after` (17:08 PT): after-hours moves of index and sector ETFs, top-300 stocks and the non-S&P list moving 2% or more after hours,
  and companies reporting earnings that day.

Sector and company names come from `agent/sectors.json` (built from yfinance).

## Usage

```bash
pip install -r requirements.txt
python -m pytest -q tests

# live report
python scripts/live_report.py                    # today's report, updates state/
python scripts/live_report.py --dry-run          # preview without changing state
python scripts/live_report.py --today 2026-10-02 --dry-run --state-dir /tmp/x   # replay a past day

# record actual fills
python scripts/record_trade.py buy MA 2 525.30   # confirm a buy (shares, price)
python scripts/record_trade.py nobuy MA          # recommended but not bought
python scripts/record_trade.py sell MA 2 540.00  # confirm a sell
python scripts/record_trade.py nosell MA         # recommended sell not executed
python scripts/record_trade.py show              # positions and cash

python scripts/market_recap.py close   # market recap after the regular close
python scripts/market_recap.py after   # after-hours recap

# research
python scripts/compare_meanrev.py                # RSI(2) / cumulative RSI(2) / IBS / EMA6-12 comparison
python scripts/research_rsi2.py                  # single technical filters on top of RSI(2)
python scripts/research_rsi2.py combos           # filter combinations x exits x sizing
python scripts/final_rsi2.py                     # year-by-year results of the selected rule
```

## Research summary (2026-10)

All tests use the point-in-time S&P 500, train 2019–2024 and test 2025-01 onward unless noted.

1. **Analyst ratings + momentum (v1, legacy).** A hand-picked list of 46 large caps looked good (+0.13R per trade),
   but on the point-in-time S&P 500 the edge fell to +0.05–0.07R and ~4% CAGR, far below SPY. An event study showed
   upgrades move the stock ~1.4% on the day, with no excess return from the next open over 1–60 days.
2. **Short-term rules.** Mean reversion (Connors RSI(2), cumulative RSI(2), IBS) gives 60–70% win rates;
   short moving-average crossovers (EMA6/EMA12) win only ~40% and lost money out of sample.
3. **Filters on RSI(2).** Of 22 filters (MACD, Bollinger Bands, ADX, stochastics, volume, gaps, VIX, ...), only
   VIX > 20, SPY RSI(2) < 30 and MACD histogram > 0 helped in the train period; the combination was selected on
   train data and then checked on the test period.
4. **Execution and costs.** Next-open execution roughly halves returns, so orders must go in before the close.
   With small accounts a US$1–2 minimum commission per order erases the edge; the strategy needs a zero-commission
   broker.
5. **Add-ons tested on the live configuration.** Market-above-200-day, gap filters, volume filters, longer/shorter
   time stops and tighter stops did not help consistently. A 10% stop or no stop scored slightly better, but the
   8% stop was kept to cap single-trade losses (no-stop worst trade: −25%).

6. **Breakout (research only, not live).** `agent/breakout.py`, `scripts/research_breakout.py`: Bollinger squeeze +
   rising 50-day SMA + contracting range with higher lows + drying volume + near the base high + stronger than SPY,
   then a close above the prior 20-day high on >= 1.2x volume. Best stable variant (breakout-day-low stop, exit below
   the 20-day SMA, top 150): 2019-24 8.8% CAGR / -9.2% max DD / 31% win rate / payoff 3.7; 2025-26 7.5% / -12.8%.
   Plain 20-day breakouts without the setup lost money in 2019-24 and only worked in 2025-26. Daily returns are
   almost uncorrelated with RSI(2) (0.14); a 50/50 mix had a lower drawdown than either alone, but both trail SPY.

7. **Look-ahead and engine audit.** `tests/test_lookahead.py` and `scripts/audit_lookahead.py` scramble all data
   after a cutoff and require identical signals, equity and trades up to it (five injected look-ahead bugs were
   all caught; the real code passes at four stress-day cutoffs). `research/nautilus/` re-runs the live RSI(2)
   rules on Nautilus Trader's event-driven engine: all 640 trades match ours on ticker, dates, entry price and
   exit reason; the only differences are 16 gap-down stops, which we fill at the open and Nautilus fills at the
   stop price (see `research/nautilus/RESULTS.md`). Both engines still assume a fill at the signal-day close.

## Limitations

- Yahoo has no prices for delisted / renamed tickers; ~85% of the 694 S&P 500 members since 2018 have data. The
  missing names bias dip-buying strategies upward.
- Ticker reuse can attach a different company's prices to an old symbol.
- The live report uses prices ~45 minutes before the close; signals near thresholds can change by the close.
- Historical earnings dates were not available (finance.yahoo.com blocked), so the earnings filter is untested
  and only used as a reminder.
- The dollar-volume ranking and 5-position sizing were chosen after seeing both periods, so the test-period
  numbers for those choices are optimistic.
- The live report aborts if fewer than 90% of current index members have data (e.g. Yahoo rate limiting).

## Repository layout

| Path | Purpose |
|---|---|
| `agent/live_rsi2.py` | Live strategy rules, model account, report rendering |
| `agent/ledger.py` | Confirm / cancel buys and sells from actual fills |
| `agent/meanrev.py` | Mean-reversion features, extended indicators, signal backtester, dollar-volume rank |
| `agent/data.py`, `agent/universe.py` | yfinance data with CSV cache; point-in-time S&P 500 membership |
| `agent/signals.py`, `strategy.py`, `backtest.py`, `report.py`, `metrics.py` | Legacy v1 analyst + momentum strategy |
| `scripts/` | Live report, trade recording, research and backtest entry points |
| `state/` | Model account (`account.json`, `positions.csv`, `trades.csv`); local only, git-ignored |
| `reports/live/` | Daily trade reports; local only, git-ignored |
| `tests/` | Unit tests (fills, look-ahead, membership, live flow, ledger) |
| `config.yaml` | Data settings and legacy v1 parameters |
