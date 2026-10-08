"""Re-run the live RSI(2) configuration on Nautilus Trader's event-driven backtest engine and reconcile it with
our own backtester (agent/meanrev.py). Run in a separate env with `pip install nautilus_trader`.

Signals (entry / exit / rank) come from prep.py and are only read for the bar being processed. Nautilus does
the execution and accounting: market orders, venue-side GTC stop orders (triggered intraday from the daily
bar's open/high/low/close path, gaps fill at the open), cash and positions.

Timing per trading day (bars are stamped 21:00 UTC, i.e. at the close):
  bar processed  -> resting stop orders may trigger
  close + 1 ns   -> exits: close > 5-day SMA or 10-day time stop -> cancel stop, market sell (fills at the close)
  close + 2 ns   -> entries: lowest RSI(2) first, <= 3 per day, <= 5 positions, ~equity/5 whole shares,
                    market buy (fills at the close), then an 8% GTC stop
"""
from __future__ import annotations

import math
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig, StrategyConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce
from nautilus_trader.model.identifiers import InstrumentId, Symbol, TraderId, Venue
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
VENUE = Venue("SIM")
EQUITY0 = 5260
MAX_POS, MAX_NEW, STOP_PCT, TIME_STOP = 5, 3, 0.08, 10


def iid(t: str) -> InstrumentId:
    return InstrumentId(Symbol(t), VENUE)


def btype(t: str) -> BarType:
    return BarType.from_str(f"{t}.SIM-1-DAY-LAST-EXTERNAL")


class RSI2Config(StrategyConfig, frozen=True):
    strategy_id: str = "RSI2-001"


class RSI2(Strategy):
    def __init__(self, sig: dict, order: dict):
        super().__init__(RSI2Config())
        self.sig = sig            # (ticker, ns) -> (entry, exit, rank, close)
        self.order = order        # ticker -> tie-break order
        self.tickers = list(order)
        self.today = None
        self.today_bars = {}      # ticker -> close of today's bar
        self.last_close = {}
        self.held = {}            # ticker -> dict(entry_ns, n, stop_order_id, qty, entry_px)
        self.equity = {}
        self.trades = []
        self.cash_shadow = None

    def on_start(self):
        for t in self.tickers:
            self.subscribe_bars(btype(t))

    def on_bar(self, bar: Bar):
        t = bar.bar_type.instrument_id.symbol.value
        if self.today != bar.ts_event:
            self.today = bar.ts_event
            self.today_bars = {}
            self.clock.set_time_alert(f"exits-{bar.ts_event}", pd.Timestamp(bar.ts_event + 1, tz="UTC"), self.do_exits)
            self.clock.set_time_alert(f"entries-{bar.ts_event}", pd.Timestamp(bar.ts_event + 2, tz="UTC"),
                                      self.do_entries)
        self.today_bars[t] = float(bar.close)
        self.last_close[t] = float(bar.close)

    def cash(self) -> float:
        return float(self.portfolio.account(VENUE).balance_total(USD).as_double())

    def do_exits(self, _):
        for t in list(self.held):
            h = self.held[t]
            if h["entry_ns"] == self.today or t not in self.today_bars:
                continue
            h["n"] += 1
            ex = self.sig[(t, self.today)][1]
            if ex or h["n"] >= TIME_STOP:
                h["reason"] = "signal" if ex else "time"
                stop = self.cache.order(h["stop_id"])
                if stop is not None and stop.is_open:
                    self.cancel_order(stop)
                self.submit_order(self.order_factory.market(iid(t), OrderSide.SELL, Quantity.from_int(h["qty"]),
                                                            reduce_only=True))

    def do_entries(self, _):
        cash = self.cash()
        equity = cash + sum(h["qty"] * self.last_close[t] for t, h in self.held.items())
        self.equity[self.today] = equity
        slots = min(MAX_POS - len(self.held), MAX_NEW)
        if slots <= 0:
            return
        cands = [t for t in self.today_bars if self.sig[(t, self.today)][0] and t not in self.held]
        cands.sort(key=lambda t: (math.isnan(self.sig[(t, self.today)][2]), self.sig[(t, self.today)][2],
                                  self.order[t]))
        for t in cands[:slots]:
            px = self.today_bars[t]
            sh = int(min(equity / MAX_POS, cash) // px)
            if sh < 1:
                continue
            cash -= sh * px
            self.held[t] = dict(entry_ns=self.today, n=0, qty=sh, stop_id=None, reason=None)
            self.submit_order(self.order_factory.market(iid(t), OrderSide.BUY, Quantity.from_int(sh)))

    def on_order_filled(self, e):
        t = e.instrument_id.symbol.value
        px = float(e.last_px)
        if e.order_side == OrderSide.BUY:
            h = self.held[t]
            h["entry_px"] = px
            trig = Price(round(px * (1 - STOP_PCT), 4), 4)
            stop = self.order_factory.stop_market(iid(t), OrderSide.SELL, Quantity.from_int(h["qty"]),
                                                  trigger_price=trig, time_in_force=TimeInForce.GTC,
                                                  reduce_only=True)
            h["stop_id"] = stop.client_order_id
            self.submit_order(stop)
        else:
            h = self.held.pop(t)
            is_stop = e.client_order_id == h["stop_id"]
            self.trades.append(dict(ticker=t, entry_date=pd.Timestamp(h["entry_ns"]).normalize(),
                                    entry=h["entry_px"], exit_date=pd.Timestamp(e.ts_event).normalize(), exit=px,
                                    ret=px / h["entry_px"] - 1, pnl=(px - h["entry_px"]) * h["qty"], days=h["n"],
                                    reason="stop" if is_stop else h["reason"], shares=h["qty"]))


def main():
    df = pd.read_csv(OUT / "bars_signals.csv", parse_dates=["date"])
    df["ns"] = (df.date + pd.Timedelta(hours=21)).astype("int64")
    order = df.drop_duplicates("ticker").set_index("ticker")["order"].to_dict()
    sig = {(t, ns): (bool(e), bool(x), float(r) if pd.notna(r) else float("nan"), c)
           for t, ns, e, x, r, c in zip(df.ticker, df.ns, df.entry, df["exit"], df["rank"], df.close)}

    eng = BacktestEngine(BacktestEngineConfig(trader_id=TraderId("BT-001"),
                                              logging=LoggingConfig(log_level="ERROR")))
    eng.add_venue(VENUE, OmsType.NETTING, AccountType.CASH, [Money(EQUITY0, USD)], base_currency=USD)
    for t, g in df.groupby("ticker"):
        eng.add_instrument(Equity(instrument_id=iid(t), raw_symbol=Symbol(t), currency=USD, price_precision=4,
                                  price_increment=Price.from_str("0.0001"), lot_size=Quantity.from_int(1),
                                  ts_event=0, ts_init=0, maker_fee=Decimal(0), taker_fee=Decimal(0)))
        bt = btype(t)
        bars = [Bar(bt, Price(o, 4), Price(h, 4), Price(l, 4), Price(c, 4), Quantity(v, 0), ns, ns)
                for o, h, l, c, v, ns in zip(g.open, g.high, g.low, g.close, g.volume, g.ns)]
        eng.add_data(bars)
    strat = RSI2(sig, order)
    eng.add_strategy(strat)
    eng.run()

    tr = pd.DataFrame(strat.trades)
    eq = pd.Series({pd.Timestamp(k).normalize(): v for k, v in strat.equity.items()}, name="equity").sort_index()
    tr.to_csv(OUT / "nautilus_trades.csv", index=False)
    eq.rename_axis("date").to_csv(OUT / "nautilus_equity.csv")
    final_cash = float(eng.portfolio.account(VENUE).balance_total(USD).as_double())
    print(f"nautilus: {len(tr)} closed trades, {len(strat.held)} open, last equity {eq.iloc[-1]:,.2f}, "
          f"cash {final_cash:,.2f}")
    eng.dispose()


if __name__ == "__main__":
    sys.exit(main())
