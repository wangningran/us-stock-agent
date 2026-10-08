"""Smoke test: how does Nautilus fill a market order submitted right after a daily bar closes?"""
from decimal import Decimal
import pandas as pd
from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType
from nautilus_trader.model.enums import AccountType, OmsType, OrderSide, TimeInForce, TriggerType
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, TraderId
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig, LoggingConfig

V = Venue("SIM")
iid = InstrumentId(Symbol("AAA"), V)
inst = Equity(instrument_id=iid, raw_symbol=Symbol("AAA"), currency=USD, price_precision=2,
              price_increment=Price.from_str("0.01"), lot_size=Quantity.from_int(1), ts_event=0, ts_init=0,
              maker_fee=Decimal(0), taker_fee=Decimal(0))
bt = BarType.from_str("AAA.SIM-1-DAY-LAST-EXTERNAL")
rows = [(100,101,99,100),(100,102,98,101),(110,111,100,105),(104,106,90,95),(96,97,94,96)]
days = pd.bdate_range("2024-01-02", periods=len(rows))
bars = []
for d,(o,h,l,c) in zip(days, rows):
    ts = int((d + pd.Timedelta(hours=21)).value)
    bars.append(Bar(bt, Price.from_str(f"{o:.2f}"), Price.from_str(f"{h:.2f}"), Price.from_str(f"{l:.2f}"),
                    Price.from_str(f"{c:.2f}"), Quantity.from_int(1000), ts, ts))

class S(Strategy):
    def on_start(self):
        self.subscribe_bars(bt); self.n = 0
    def on_bar(self, bar):
        self.n += 1
        self.log.warning(f"bar {self.n} close={bar.close}")
        if self.n == 2:
            self.clock.set_time_alert("buy", pd.Timestamp(bar.ts_event + 1, tz="UTC"), self.buy)
    def buy(self, ev):
        o = self.order_factory.market(iid, OrderSide.BUY, Quantity.from_int(10))
        self.submit_order(o)
        st = self.order_factory.stop_market(iid, OrderSide.SELL, Quantity.from_int(10), trigger_price=Price.from_str("92.00"),
                                            time_in_force=TimeInForce.GTC)
        self.submit_order(st)
    def on_order_filled(self, e):
        self.log.warning(f"FILL {e.order_side} {e.last_qty} @ {e.last_px} ts={pd.Timestamp(e.ts_event, tz='UTC')}")

eng = BacktestEngine(BacktestEngineConfig(trader_id=TraderId("T-001"), logging=LoggingConfig(log_level="WARNING")))
eng.add_venue(V, OmsType.NETTING, AccountType.CASH, [Money(10000, USD)], base_currency=USD)
eng.add_instrument(inst); eng.add_data(bars); eng.add_strategy(S(StrategyConfig(strategy_id="S-001")))
eng.run()
acct = eng.portfolio.account(V); print("balance", acct.balance_total(USD))
