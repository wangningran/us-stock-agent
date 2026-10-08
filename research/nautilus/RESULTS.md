# Nautilus Trader cross-check: live RSI(2) configuration

Same data (Yahoo daily bars rounded to 4 dp, point-in-time S&P 500), same signals, US$5,260, whole shares,
zero commission and zero slippage in both engines, 2019-01-02 to the latest bar.

## Trade-by-trade reconciliation

- Trades: ours 640, Nautilus 640; matched on ticker + entry date: 640, unmatched: 0
- Entry prices identical: yes; exit dates identical: yes; exit reasons identical: yes
- Exit prices differ on 16 trades, all stop exits on a gap-down open (confirmed): we fill at the open (below the stop), Nautilus's bar execution fills at the stop price. Ours is the conservative / realistic assumption.
- Equity curves before the first gap stop (2019-08-05): max difference 0.0002% (Nautilus keeps cash in cents).

## Results

| Period | Engine | Trades | Win rate | CAGR | Max DD | Sharpe |
|---|---|---|---|---|---|---|
| 2019-2024 | ours | 511 | 72.8% | +11.7% | -13.6% | 1.34 |
| 2019-2024 | Nautilus | 511 | 72.8% | +12.3% | -12.8% | 1.42 |
| 2025-2026 | ours | 129 | 73.6% | +14.7% | -4.7% | 1.53 |
| 2025-2026 | Nautilus | 129 | 73.6% | +15.6% | -4.3% | 1.60 |
| 2019-2026 | ours | 640 | 73.0% | +12.4% | -13.6% | 1.39 |
| 2019-2026 | Nautilus | 640 | 73.0% | +13.0% | -12.8% | 1.47 |

## Notes

- Both engines assume the signal is computed at the close and the order fills at that close (market-on-close). Nautilus fills a market order sent right after a daily bar at that bar's close, so it does not remove this assumption; in live trading the signal uses prices ~45 minutes before the close.
- Nautilus's cash-account risk engine denies sell stops when free cash is low unless they are reduce-only; the run uses reduce-only exits.
- Data-level biases (missing delisted tickers, adjusted prices) are the same in both engines.
