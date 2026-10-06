"""Market recap reports (Chinese, user-facing): regular-session close and after-hours close.

Covers indices / macro, breadth, sectors with their leaders, notable movers, strategy watch-list and earnings.
"""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .meanrev import wilder_rsi

INDEX = {"SPY": "标普500", "QQQ": "纳指100", "IWM": "罗素2000", "DIA": "道指"}
MACRO = {"^VIX": "VIX 恐慌指数", "^TNX": "10年美债收益率", "DX-Y.NYB": "美元指数", "GLD": "黄金",
         "USO": "原油", "IBIT": "比特币 ETF"}
SECTOR_ETF = {"Technology": ("XLK", "科技"), "Communication Services": ("XLC", "通信"),
              "Consumer Cyclical": ("XLY", "可选消费"), "Financial Services": ("XLF", "金融"),
              "Healthcare": ("XLV", "医疗"), "Industrials": ("XLI", "工业"), "Energy": ("XLE", "能源"),
              "Consumer Defensive": ("XLP", "必选消费"), "Utilities": ("XLU", "公用事业"),
              "Basic Materials": ("XLB", "材料"), "Real Estate": ("XLRE", "地产")}
EXTRA_ETF = {"SMH": "半导体"}
# Heavily traded US-listed stocks outside the S&P 500 (new IPOs, ADRs, ...); ~US$1B+ average daily dollar volume.
HOT_EXTRA = {"SPCX": "SpaceX", "TSM": "台积电", "ASML": "阿斯麦", "MSTR": "Strategy（比特币）", "ARM": "Arm",
             "SNOW": "Snowflake", "RKLB": "Rocket Lab", "SHOP": "Shopify", "BABA": "阿里巴巴", "NU": "Nu Holdings",
             "CRCL": "Circle"}
SECTORS_FILE = Path(__file__).with_name("sectors.json")


def _quiet():
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def load_meta() -> dict:
    return json.loads(SECTORS_FILE.read_text()) if SECTORS_FILE.exists() else {}


def download_daily(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    import yfinance as yf
    _quiet()
    raw = yf.download(tickers, period=period, auto_adjust=True, progress=False, threads=True, group_by="column")
    _fill_last_bar(raw, tickers)
    out = {}
    for t in tickers:
        try:
            df = pd.DataFrame({k.lower(): raw[k][t] for k in ("Open", "High", "Low", "Close", "Volume")}).dropna(
                subset=["close"])
        except KeyError:
            continue
        if len(df):
            df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
            out[t] = df
    return out


def _fill_last_bar(raw: pd.DataFrame, tickers: list[str]) -> None:
    """Yahoo sometimes leaves today's daily close empty for hours after the close (open/volume present).
    Fill open/high/low/close of that last row from today's regular-session 5-minute bars."""
    import yfinance as yf
    if not isinstance(raw.columns, pd.MultiIndex) or raw.empty:
        return
    last = raw.index[-1]
    missing = [t for t in tickers if t in raw["Close"] and pd.isna(raw["Close"][t].iloc[-1])
               and pd.notna(raw["Volume"][t].iloc[-1])]
    if not missing:
        return
    intra = yf.download(missing, period="1d", interval="5m", prepost=False, progress=False, threads=True,
                        auto_adjust=True, group_by="column")
    if intra.empty or not isinstance(intra.columns, pd.MultiIndex):
        return
    day = intra.index[-1].tz_convert("America/New_York").normalize().tz_localize(None)
    if day != pd.Timestamp(last).tz_localize(None).normalize():
        return
    for t in missing:
        if t not in intra["Close"]:
            continue
        c = intra["Close"][t].dropna()
        if not len(c):
            continue
        raw.loc[last, ("Close", t)] = float(c.iloc[-1])
        raw.loc[last, ("Open", t)] = float(intra["Open"][t].dropna().iloc[0])
        raw.loc[last, ("High", t)] = float(intra["High"][t].max())
        raw.loc[last, ("Low", t)] = float(intra["Low"][t].min())


def download_afterhours(tickers: list[str]) -> pd.DataFrame:
    """Latest post-market price per ticker (16:00-20:00 ET) and the regular-session close from intraday bars."""
    import yfinance as yf
    _quiet()
    raw = yf.download(tickers, period="1d", interval="5m", prepost=True, progress=False, threads=True,
                      auto_adjust=False, group_by="column")
    c = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]].rename(columns={"Close": tickers[0]})
    et = c.index.tz_convert("America/New_York")
    post = c[et.time >= pd.Timestamp("16:00").time()]
    rows = {}
    for t in c.columns:
        p = post[t].dropna()
        if len(p):
            rows[t] = {"ah_last": float(p.iloc[-1]), "ah_time": p.index[-1].tz_convert("America/New_York").strftime("%H:%M")}
    return pd.DataFrame(rows).T


def earnings_dates(tickers: list[str], workers: int = 8) -> dict[str, pd.Timestamp]:
    import yfinance as yf
    _quiet()

    def one(t):
        try:
            d = (yf.Ticker(t).calendar or {}).get("Earnings Date") or []
            return t, [pd.Timestamp(x) for x in d]
        except Exception:  # noqa: BLE001
            return t, []
    with ThreadPoolExecutor(workers) as ex:
        return {t: d for t, d in ex.map(one, tickers) if d}


def _chg(df: pd.DataFrame, n: int) -> float:
    return float(df.close.iloc[-1] / df.close.iloc[-1 - n] - 1) if len(df) > n else np.nan


def _ytd(df: pd.DataFrame) -> float:
    y = df[df.index >= pd.Timestamp(f"{df.index[-1].year}-01-01")]
    return float(df.close.iloc[-1] / y.close.iloc[0] - 1) if len(y) else np.nan


def _pct(x, sign=True):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else (f"{x:+.2%}" if sign else f"{x:.2%}")


def stock_table(daily: dict[str, pd.DataFrame], tickers: list[str], day: pd.Timestamp, meta: dict) -> pd.DataFrame:
    rows = []
    for t in tickers:
        d = daily.get(t)
        if d is None or d.index[-1] != day or len(d) < 30:
            continue
        c = d.close
        dv = (c * d.volume).rolling(60, min_periods=20).mean().iloc[-1]
        rows.append({
            "ticker": t, "name": meta.get(t, {}).get("name", t), "sector": meta.get(t, {}).get("sector", ""),
            "close": float(c.iloc[-1]), "ret1": _chg(d, 1), "ret5": _chg(d, 5),
            "vol_ratio": float(d.volume.iloc[-1] / d.volume.iloc[-21:-1].mean()) if len(d) > 21 else np.nan,
            "dv": float(dv), "rsi2": float(wilder_rsi(c, 2).iloc[-1]),
            "above200": bool(len(c) >= 200 and c.iloc[-1] > c.rolling(200).mean().iloc[-1]),
            "hi52": bool(d.high.iloc[-1] >= d.high.iloc[-252:].max() * 0.999),
            "lo52": bool(d.low.iloc[-1] <= d.low.iloc[-252:].min() * 1.001),
            "macd_hist": float((lambda m: m - m.ewm(span=9, adjust=False).mean())(
                c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()).iloc[-1]),
        })
    df = pd.DataFrame(rows)
    df["dv_rank"] = df.dv.rank(ascending=False)
    return df


def _short(name: str, n: int = 22) -> str:
    for s in (" Corporation", " Incorporated", " Inc.", " Inc", " Corp.", " Co.", " Company", " Holdings", " plc",
              " Ltd.", ", Inc.", " Common Stock", " Common St", " Class A", " (The)"):
        name = name.replace(s, "")
    return name.strip(" ,")[:n]


def build_close(today: pd.Timestamp, members: list[str]) -> str:
    meta = load_meta()
    etfs = list(INDEX) + list(MACRO) + [v[0] for v in SECTOR_ETF.values()] + list(EXTRA_ETF)
    extras = [t for t in HOT_EXTRA if t not in members]
    daily = download_daily(etfs + members + extras)
    day = daily["SPY"].index[-1]
    if day != today:
        return f"# {today:%Y-%m-%d} 美股休市\n\n最新交易日为 {day:%Y-%m-%d}，今天没有收盘报告。\n"
    st = stock_table(daily, members, day, meta)
    L = [f"# 📈 美股收盘报告 · {day:%Y-%m-%d}", ""]

    # summary line
    idx_chg = {n: _chg(daily[t], 1) for t, n in INDEX.items()}
    sec = []
    for s, (etf, cn) in SECTOR_ETF.items():
        if etf in daily:
            sec.append((cn, etf, s, _chg(daily[etf], 1), _chg(daily[etf], 5)))
    sec.sort(key=lambda x: -x[3])
    up_n = int((st.ret1 > 0).sum())
    lead = max(idx_chg, key=idx_chg.get)
    tone = "齐涨" if all(v > 0 for v in idx_chg.values()) else "齐跌" if all(v < 0 for v in idx_chg.values()) else "涨跌互现"
    vix = daily["^VIX"].close.iloc[-1]
    L.append(f"**一句话**：主要指数{tone}，{lead}表现最好（{_pct(idx_chg[lead])}）；"
             f"{sec[0][0]}领涨（{_pct(sec[0][3])}），{sec[-1][0]}最弱（{_pct(sec[-1][3])}）；"
             f"标普500成分股 {up_n}/{len(st)} 上涨，VIX {vix:.1f}。")
    L.append("")

    # indices and macro
    L.append("## 一、大盘与宏观")
    L.append("| 品种 | 收盘 | 今日 | 5日 | 年初至今 | RSI(2) |")
    L.append("|---|---|---|---|---|---|")
    for t, n in {**INDEX, **MACRO}.items():
        d = daily.get(t)
        if d is None:
            continue
        L.append(f"| {n}（{t.lstrip('^')}） | {d.close.iloc[-1]:,.2f} | {_pct(_chg(d, 1))} | {_pct(_chg(d, 5))} "
                 f"| {_pct(_ytd(d))} | {wilder_rsi(d.close, 2).iloc[-1]:.0f} |")
    L.append("")

    # breadth
    L.append("## 二、市场宽度（标普500成分股）")
    L.append(f"- 上涨 **{up_n}** 家 / 下跌 **{int((st.ret1 < 0).sum())}** 家，涨跌幅中位数 {_pct(st.ret1.median())}")
    L.append(f"- 52 周新高 **{int(st.hi52.sum())}** 家 / 新低 **{int(st.lo52.sum())}** 家；"
             f"站上 200 日线 **{st.above200.mean():.0%}**")
    L.append("")

    # sectors with leaders
    L.append("## 三、板块与龙头")
    L.append("| 板块 | 今日 | 5日 | 龙头（成交额最大） | 板块内最强 | 板块内最弱 |")
    L.append("|---|---|---|---|---|---|")
    for cn, etf, s, d1, d5 in sec:
        g = st[st.sector == s]
        if g.empty:
            continue
        leaders = g.nsmallest(2, "dv_rank")
        best, worst = g.loc[g.ret1.idxmax()], g.loc[g.ret1.idxmin()]
        lead_s = "、".join(f"{r.ticker} {_pct(r.ret1)}" for r in leaders.itertuples())
        L.append(f"| {cn}（{etf}） | {_pct(d1)} | {_pct(d5)} | {lead_s} | {best.ticker} {_pct(best.ret1)} "
                 f"| {worst.ticker} {_pct(worst.ret1)} |")
    if "SMH" in daily:
        L.append(f"| 半导体（SMH） | {_pct(_chg(daily['SMH'], 1))} | {_pct(_chg(daily['SMH'], 5))} | — | — | — |")
    L.append("")

    # notable movers
    L.append("## 四、涨跌幅明显的个股（标普500，|涨跌| ≥ 4%）")
    for title, sub in (("涨幅", st[st.ret1 >= 0.04].nlargest(10, "ret1")),
                       ("跌幅", st[st.ret1 <= -0.04].nsmallest(10, "ret1"))):
        if sub.empty:
            L.append(f"- {title}：无")
            continue
        L.append("")
        L.append(f"**{title}**")
        L.append("| 股票 | 公司 | 板块 | 今日 | 5日 | 成交量/20日均量 |")
        L.append("|---|---|---|---|---|---|")
        for r in sub.itertuples():
            cn = SECTOR_ETF.get(r.sector, ("", r.sector))[1]
            L.append(f"| {r.ticker} | {_short(r.name)} | {cn} | {_pct(r.ret1)} | {_pct(r.ret5)} | {r.vol_ratio:.1f}x |")
    hot = [(t, daily[t]) for t in extras if t in daily and daily[t].index[-1] == day]
    if hot:
        hot.sort(key=lambda x: -_chg(x[1], 1))
        L.append("")
        L.append("**热门非标普股票**（不在标普500里、但成交额很大，⚡ 表示 |涨跌| ≥ 4%）")
        L.append("| 股票 | 公司 | 收盘 | 今日 | 5日 |")
        L.append("|---|---|---|---|---|")
        for t, d in hot:
            r1 = _chg(d, 1)
            L.append(f"| {t}{' ⚡' if abs(r1) >= 0.04 else ''} | {HOT_EXTRA[t]} | {d.close.iloc[-1]:,.2f} "
                     f"| {_pct(r1)} | {_pct(_chg(d, 5))} |")
    L.append("")

    # strategy watch
    pool = st[st.dv_rank <= 150]
    spy_r2 = wilder_rsi(daily["SPY"].close, 2).iloc[-1]
    stress = vix > 20 or spy_r2 < 30
    L.append("## 五、策略观察（RSI(2) 均值回归）")
    L.append(f"- 市场压力：VIX {vix:.1f}（需 > 20）、SPY RSI(2) {spy_r2:.0f}（需 < 30）→ "
             + ("✅ **满足**，明天有机会开新仓" if stress else "⏸ 不满足，股票不开新仓"))
    near = pool[pool.above200 & (pool.rsi2 < 15)].sort_values("rsi2")
    if near.empty:
        L.append("- 股票池里没有接近超卖的股票")
    else:
        full = near[(near.rsi2 < 10) & (near.macd_hist > 0)]
        L.append(f"- 接近信号（股票池内、200 日线上方、RSI(2) < 15）：" + "、".join(
            f"{r.ticker} {r.rsi2:.1f}" for r in near.head(10).itertuples()))
        if len(full):
            L.append(f"- **个股条件全部满足、只差市场压力**：{'、'.join(full.ticker)}")
    if "IBIT" in daily:
        d = daily["IBIT"]
        r2 = wilder_rsi(d.close, 2).iloc[-1]
        L.append(f"- 比特币 ETF（IBIT）：RSI(2) {r2:.0f}，{'200 日线上方' if d.close.iloc[-1] > d.close.rolling(200).mean().iloc[-1] else '200 日线下方'}"
                 + ("，✅ 满足买入条件" if r2 < 10 and d.close.iloc[-1] > d.close.rolling(200).mean().iloc[-1] else ""))
    L.append("")

    # earnings next trading day
    top = st.nsmallest(200, "dv_rank").ticker.tolist()
    nxt = pd.bdate_range(day + pd.Timedelta(days=1), periods=1)[0]
    ed = earnings_dates(top)
    tom = [t for t, ds in ed.items() if any(x.normalize() == nxt for x in ds)]
    L.append(f"## 六、下一交易日（{nxt:%m-%d}）财报")
    L.append("、".join(f"{t}（{_short(meta.get(t, {}).get('name', t), 16)}）" for t in tom) if tom else "成交额前 200 的公司中没有。")
    L.append("")
    L.append("> 数据来自 Yahoo Finance 收盘数据，仅供研究参考，不构成投资建议。")
    return "\n".join(L)


def build_afterhours(today: pd.Timestamp, members: list[str]) -> str:
    meta = load_meta()
    etfs = list(INDEX) + [v[0] for v in SECTOR_ETF.values()] + ["SMH", "IBIT"]
    daily = download_daily(etfs + members, period="3mo")
    day = daily["SPY"].index[-1]
    if day != today:
        return f"# {today:%Y-%m-%d} 美股休市\n\n今天没有盘后报告。\n"
    st = stock_table(daily, members, day, meta)
    top = st.nsmallest(300, "dv_rank").ticker.tolist()
    extras = [t for t in HOT_EXTRA if t not in members]
    extra_daily = download_daily(extras, period="1mo")
    daily.update({t: d for t, d in extra_daily.items() if d.index[-1] == day})
    top += [t for t in extras if t in daily]
    for t in extras:
        meta.setdefault(t, {"name": HOT_EXTRA[t], "sector": ""})
    ah = download_afterhours(etfs + top)
    close = {t: float(daily[t].close.iloc[-1]) for t in etfs + top if t in daily}
    ah["close"] = pd.Series(close)
    ah = ah.dropna(subset=["close"])
    ah["ah_chg"] = ah.ah_last / ah.close - 1
    ed = earnings_dates(top)
    reporting = {t for t, ds in ed.items() if any(x.normalize() == day for x in ds)}
    asof = ah.ah_time.max() if len(ah) else "—"

    L = [f"# 🌙 美股盘后报告 · {day:%Y-%m-%d}（盘后数据截至美东 {asof}）", ""]
    idx_rows = [(INDEX[t], t, ah.loc[t]) for t in INDEX if t in ah.index]
    if idx_rows:
        moves = "，".join(f"{n} {_pct(r.ah_chg)}" for n, t, r in idx_rows)
        L.append(f"**一句话**：盘后 {moves}。")
        L.append("")
    L.append("## 一、指数 ETF 盘后")
    L.append("| 品种 | 收盘价 | 盘后最新 | 盘后涨跌 |")
    L.append("|---|---|---|---|")
    for n, t, r in idx_rows:
        L.append(f"| {n}（{t}） | {r.close:.2f} | {r.ah_last:.2f} | {_pct(r.ah_chg)} |")
    for t, n in (("SMH", "半导体"), ("IBIT", "比特币 ETF")):
        if t in ah.index:
            r = ah.loc[t]
            L.append(f"| {n}（{t}） | {r.close:.2f} | {r.ah_last:.2f} | {_pct(r.ah_chg)} |")
    L.append("")

    L.append("## 二、板块 ETF 盘后")
    sec = [(cn, etf, ah.loc[etf].ah_chg) for s, (etf, cn) in SECTOR_ETF.items() if etf in ah.index]
    sec.sort(key=lambda x: -x[2])
    L.append("、".join(f"{cn} {_pct(c)}" for cn, etf, c in sec) if sec else "板块 ETF 盘后几乎没有成交。")
    L.append("")

    L.append("## 三、盘后异动个股（成交额前 300 + 热门非标普，|盘后涨跌| ≥ 2%）")
    movers = ah.loc[[t for t in top if t in ah.index]]
    movers = movers[movers.ah_chg.abs() >= 0.02].sort_values("ah_chg", key=lambda s: -s.abs()).head(15)
    if movers.empty:
        L.append("没有明显异动。")
    else:
        L.append("| 股票 | 公司 | 板块 | 收盘价 | 盘后最新 | 盘后涨跌 | 备注 |")
        L.append("|---|---|---|---|---|---|---|")
        for t, r in movers.iterrows():
            m = meta.get(t, {})
            cn = SECTOR_ETF.get(m.get("sector", ""), ("", m.get("sector", "")))[1]
            L.append(f"| {t} | {_short(m.get('name', t))} | {cn} | {r.close:.2f} | {r.ah_last:.2f} | {_pct(r.ah_chg)} "
                     f"| {'📊 今日财报' if t in reporting else ''} |")
    L.append("")

    L.append("## 四、今日发布财报的公司（成交额前 300）")
    if reporting:
        parts = []
        for t in sorted(reporting, key=lambda x: st.set_index("ticker").dv_rank.get(x, 999)):
            chg = ah.ah_chg.get(t, np.nan)
            parts.append(f"{t}（{_short(meta.get(t, {}).get('name', t), 16)}，盘后 {_pct(chg)}）")
        L.append("、".join(parts))
    else:
        L.append("没有。")
    L.append("")
    L.append("> 盘后成交量小、价格波动大，仅作参考；次日开盘可能明显不同。数据来自 Yahoo Finance，不构成投资建议。")
    return "\n".join(L)
