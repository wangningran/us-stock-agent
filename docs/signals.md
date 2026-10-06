# Signal Reference

Every price/volume signal used or tested in this project, what it measures and how it is used.
All indicators use **daily bars** of the regular session (09:30–16:00 ET); pre-/after-market trades are ignored.

## 1. RSI(2) strategy (live, daily trade report at 12:12 PT)

| Signal | Calculation | Meaning | Use |
|---|---|---|---|
| **200-day SMA** | Average close of the last 200 trading days | Long-term trend; price above it = long-term uptrend | Buy only above it (buy short-term dips in long-term uptrends) |
| **RSI(2)** | Wilder RSI over 2 days, 0–100 | Below 10 = sharp short-term sell-off (oversold) | **Main entry**: RSI(2) < 10 |
| **MACD histogram** | (12-day EMA − 26-day EMA) − its 9-day EMA | Positive = momentum improving; negative = decline accelerating | Must be > 0 (avoid "catching a falling knife") |
| **VIX** | Implied volatility of S&P 500 options ("fear index") | Above 20 = market under stress | Market-stress condition (either this or the next one) |
| **SPY RSI(2)** | RSI(2) of the S&P 500 ETF | Below 30 = the whole market is sharply oversold | Market-stress condition |
| **Dollar-volume rank** | 60-day average of price × volume, ranked within the S&P 500 | Size / popularity, point-in-time | Only the top 150 are eligible |
| **5-day SMA** | Average close of the last 5 days | Short-term average price | **Exit**: sell when the close is above it (same as close > average of the prior 4 closes — the report's "sell line") |
| **8% stop** | Entry price × 0.92 | Caps the loss per trade | GTC stop order after each buy |
| **10-day time stop** | Trading days held | No bounce = the trade idea failed | Sell after 10 trading days |
| **IBIT (Bitcoin ETF)** | 200-day SMA + RSI(2) < 10 only | Bitcoin pullbacks | No stress / MACD / dollar-volume filters |

## 2. Breakout strategy (research, watch-only, section 6 of the close recap)

**A. Watch-list** — all of the following:

| Signal | Calculation | Meaning |
|---|---|---|
| **Bollinger squeeze** | Band width = 4 × 20-day std ÷ 20-day SMA, in the narrowest 20% of the last 60 days | Volatility contracted; often precedes a large move |
| **Above a rising 50-day SMA** | Close > 50-day SMA and the SMA is higher than 10 days ago | Medium-term uptrend |
| **Contracting range** | High–low range of the last 20 days < range of the 20 days before | The base is tightening |
| **Higher lows** | Lowest low of the last 10 days > lowest low of days 11–30 | Buyers step in at higher prices; selling pressure fading |
| **Drying volume** | 5-day average volume < 20-day average volume | Sellers exhausted |
| **Near the base high** | Close within 3% below the 20-day high | Breakout could come any day |
| **Relative strength** | 3-month return − SPY 3-month return > 0 | Stronger than the market |

**B. Breakout signal and exits:**

| Signal | Meaning |
|---|---|
| **Close above the prior 20-day high** | Price breaks out of the base |
| **Volume ratio ≥ 1.2** | Today's volume ÷ prior 20-day average; real buying behind the move |
| **Setup within the last 5 days** | The stock was on the watch-list before breaking out |
| **SPY above its 200-day SMA** | Only trade breakouts in a bull market |
| **Stop = breakout-day low** (max 10%) | Falling back below that bar = failed breakout |
| **Exit = close below the 20-day SMA** | Let winners run; leave when the trend breaks |

Backtest (2019–2026, top 150): ~31% win rate, payoff ~3.4, ~8.5% CAGR, −12.8% max drawdown.

## 3. Market recap indicators

| Indicator | Meaning |
|---|---|
| **52-week highs / lows count** | More new highs = stronger market |
| **% of stocks above the 200-day SMA** | Market health; > 60% strong, < 40% weak |
| **Advancers / decliners, median change** | Breadth: are a few giants lifting the index or most stocks rising? |
| **Volume ratio (volume / 20-day average)** | > 1.5 = unusual activity, usually news-driven |
| **5-day and year-to-date change** | Short- and medium-term strength |

## 4. Tested on top of RSI(2) but not adopted

| Signal | Meaning | Result |
|---|---|---|
| Below the lower Bollinger band / %B < 0.1 | Price at the bottom of its band (oversold) | No improvement |
| MACD line > 0 | 12-day EMA above 26-day EMA | Weaker than MACD histogram > 0 |
| ADX > 25 / ADX < 20 | Trend strength (> 25 trending, < 20 ranging) | No consistent improvement |
| Close > 50-day SMA | Medium-term uptrend | No consistent improvement |
| EMA6 > EMA12 | Short moving-average crossover | ~40% win rate as a standalone strategy; lost money out of sample |
| Stochastic %K < 20 | Price near its 14-day low | Redundant with RSI(2) |
| Volume > 1.5× / < 1× average | Panic selling vs quiet pullback | No consistent improvement |
| Gap down > 3% / 1-day drop > 5% | News-driven crash | Excluding them did not help |
| ATR% < 3% | Average true range as % of price (daily volatility) | No consistent improvement |
| > 5% below the 10-day high | Deep enough pullback | No consistent improvement |
| ≥ 3 down days in a row | Consecutive declines | No consistent improvement |
| SPY above its 200- / 50-day SMA | Bull market | Lowered returns (stress is when dips pay best) |
| VIX < 25 | Calm market | Wrong direction; VIX > 20 is what helps |
| IBS, cumulative RSI(2) | Alternative oversold measures | Similar to RSI(2); RSI(2) kept |

Only **VIX > 20, SPY RSI(2) < 30 and MACD histogram > 0** made it into the live rule.
Trade-off of the MACD filter (all S&P 500 members): 2019–24 CAGR 12.2% vs 15.2% without it, but a smaller
drawdown (−15.7% vs −18.5%); in 2025–26 the skipped trades averaged 0.0% with a −27% worst trade.

## 5. Plain-language glossary

- **SMA (simple moving average)**: average close of the last N days. Price above = strong, below = weak; larger N = longer trend.
- **EMA (exponential moving average)**: an average that weights recent prices more, so it reacts faster than an SMA.
- **RSI (relative strength index)**: share of up-moves in all moves over a window. RSI(2) uses 2 days and is very sensitive; the common RSI(14) is smoother.
- **Bollinger Bands**: 20-day SMA ± 2 standard deviations. Wide bands = high volatility; narrow bands = coiling.
- **MACD**: gap between a short and a long EMA, showing direction and strength of momentum. The histogram is the MACD line minus its 9-day signal line.
- **ATR (average true range)**: average daily price range; often used to size stops.
- **Volume ratio**: today's volume as a multiple of normal volume; flags unusual buying or selling.
