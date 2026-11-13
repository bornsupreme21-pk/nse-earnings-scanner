import os
import datetime
import requests
import pandas as pd
import yfinance as yf
from tradingview_screener import Query, Column

# --- Core V7.0 "Giant Killer" Parameters ---
MIN_PRICE_FILTER = 50.0
MIN_RSI = 44.0
MAX_RSI = 70.0           # Strictly buy before overbought euphoria
MAX_1W_CHASE_PCT = 7.5   # Never chase stocks already up >7.5% this week
LIBRARY_MAX_DAYS = 35

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*"
}

def safe_float(val, default=None):
    try:
        if pd.isna(val) or val is None:
            return default
        return float(val)
    except Exception:
        return default

def first_valid_float(row, keys, default=None):
    for k in keys:
        v = safe_float(row.get(k))
        if v is not None:
            return v
    return default

def fetch_nse_corporate_catalysts():
    """Scrapes NSE Official Board Meetings for Results, M&A, Partnerships, Buybacks, Dividends."""
    catalysts = {}
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=8)
        res = session.get("https://www.nseindia.com/api/corporate-board-meetings?index=equities", headers=NSE_HEADERS, timeout=10)
        if res.status_code == 200:
            for item in res.json():
                sym = str(item.get("bm_symbol", "")).strip()
                purpose = str(item.get("bm_purpose", "")).strip()
                date_str = str(item.get("bm_date", "")).strip()
                if sym and date_str:
                    try:
                        dt = datetime.datetime.strptime(date_str, "%d-%b-%Y").date()
                        tag = "📅 Board Meeting"
                        if "Result" in purpose: tag = "📊 Qtr Results BM"
                        elif any(k in purpose.lower() for k in ["acqui", "amalga", "merger", "scheme", "joint"]): tag = "🤝 M&A / Partnership"
                        elif any(k in purpose.lower() for k in ["fund", "preferential", "qip", "rights"]): tag = "💰 Capital Raise BM"
                        elif any(k in purpose.lower() for k in ["dividend", "buyback", "bonus", "split"]): tag = "🎁 Corporate Action BM"
                        catalysts[sym] = (dt, tag)
                    except Exception:
                        pass
    except Exception:
        pass
    return catalysts

def parse_earnings_date(ts_val, ticker, nse_catalysts, today):
    if ticker in nse_catalysts:
        dt, tag = nse_catalysts[ticker]
        days = (dt - today).days
        if days >= 0:
            return f"{dt} ({tag})", int(days)

    if pd.isna(ts_val) or ts_val is None:
        return "Unknown", ""
    try:
        e_date = datetime.datetime.fromtimestamp(int(ts_val)).date()
        days_left = (e_date - today).days
        return str(e_date), int(days_left)
    except Exception:
        return "Unknown", ""

def enrich_top_candidates_with_yfinance(records):
    """Enriches top candidates with Yahoo Finance forwardEps & analyst target prices."""
    for rec in records[:25]:
        if rec['Est Next EPS Gr (%)'] == "" or rec['Earnings Date'] == "Unknown":
            try:
                tk = yf.Ticker(f"{rec['Ticker']}.NS")
                info = tk.info or {}
                fwd_eps = safe_float(info.get('forwardEps'))
                trl_eps = safe_float(info.get('trailingEps'))
                if rec['Est Next EPS Gr (%)'] == "" and fwd_eps and trl_eps and trl_eps > 0:
                    rec['Est Next EPS Gr (%)'] = round(((fwd_eps - trl_eps) / abs(trl_eps)) * 100, 2)
                tgt_mean = safe_float(info.get('targetMeanPrice'))
                if tgt_mean and tgt_mean > rec['Min Target']:
                    rec['Max Target'] = round(max(rec['Max Target'], min(tgt_mean, rec['CMP'] * 1.25)), 2)
                    rec['Max Growth (%)'] = round(((rec['Max Target'] - rec['CMP']) / rec['CMP']) * 100, 2)
            except Exception:
                continue

def evaluate_giant_killer_score(
    sector, rsi, vol_mult, dist_52w, chg_1d, chg_1w, chg_1m,
    eff_eps_gr, qtr_rev_gr, qtr_eps_gr, op_margin, roe, pe, pb, debt_eq,
    adx, days_left, cmp, ema20, ema50
):
    score = 0
    is_financial = any(w in sector.lower() for w in ['finance', 'bank', 'insurance', 'investment'])

    # 1. Low-Phase Base Proximity (Max 25 pts) - Highest weight on buying at the quiet support floor
    dist_ema20_pct = ((cmp - ema20) / ema20) * 100 if ema20 > 0 else 0
    if -2.5 <= dist_ema20_pct <= 2.2 and -3.5 <= chg_1w <= 3.5:
        score += 25  # Coiling tightly at 20-EMA floor
    elif -4.0 <= dist_ema20_pct <= 4.0 and chg_1w <= 6.0:
        score += 18
    elif dist_ema20_pct <= 5.5:
        score += 10

    # 2. Pre-Breakout RSI & Institutional Accumulation Volume (Max 18 pts)
    if 46 <= rsi <= 62: score += 10
    elif 62 < rsi <= 68: score += 6
    elif 44 <= rsi < 46: score += 5

    if vol_mult >= 2.0: score += 8
    elif vol_mult >= 1.35: score += 6
    elif vol_mult >= 1.05: score += 4

    # 3. Consensus Forward EPS / Profit Acceleration (Max 32 pts)
    missing_fundy = 0
    if eff_eps_gr is not None:
        if eff_eps_gr >= 30: score += 18
        elif eff_eps_gr >= 18: score += 14
        elif eff_eps_gr >= 10: score += 9
        elif eff_eps_gr > 0: score += 5
    else:
        missing_fundy += 1

    if qtr_rev_gr is not None:
        if qtr_rev_gr >= 20: score += 14
        elif qtr_rev_gr >= 12: score += 10
        elif qtr_rev_gr >= 5: score += 6
    else:
        missing_fundy += 1

    # 4. Sector-Aware Balance Sheet & ROE Power (Max 15 pts)
    if is_financial:
        if (roe or 0) >= 15: score += 9
        elif (roe or 0) >= 10: score += 5
        else: missing_fundy += 1
        if pb is not None and 0 < pb <= 3.8: score += 6
        elif pe is not None and 0 < pe <= 22: score += 4
    else:
        if (op_margin or 0) >= 16: score += 6
        elif (op_margin or 0) >= 10: score += 4
        if (roe or 0) >= 16: score += 6
        elif (roe or 0) >= 11: score += 4
        else: missing_fundy += 1
        if debt_eq is not None and debt_eq <= 0.6: score += 3

    # 5. Pre-Catalyst Window (Max 10 pts)
    regime = "💎 LOW-PHASE BASE BUY"
    if isinstance(days_left, int):
        if 0 <= days_left <= 1:
            regime = "⛔ BINARY EVENT (Do Not Enter)"
            score -= 25
        elif 2 <= days_left <= 15:
            score += 10
            regime = "💎 PRE-EVENT LOW-PHASE BUY (Exit T-1)" if dist_ema20_pct <= 2.2 else "🎯 PRE-ER RUNUP (Exit T-1)"
        elif 16 <= days_left <= 35:
            score += 9
            regime = "⏳ EARLY PRE-RESULT ACCUMULATION"
        elif 65 <= days_left <= 92 and (qtr_eps_gr or 0) >= 20 and dist_ema20_pct <= 3.0:
            score += 7
            regime = "⚡ POST-RESULT PULLBACK BUY"
        else:
            score += 4
    else:
        if (eff_eps_gr or 0) >= 20 and dist_ema20_pct <= 2.5:
            score += 7
            regime = "🔮 CONSENSUS BEAT ACCUMULATION"

    if missing_fundy >= 2:
        score = min(score, 72)

    score = min(max(int(round(score)), 0), 100)

    if "BINARY EVENT" in regime:
        decision = "⛔ AVOID EVENT GAP"
    elif score >= 84:
        decision = "🚀 GIANT-KILLER BUY"
    elif score >= 75:
        decision = "🔥 STRONG ACCUMULATE"
    elif score >= 65:
        decision = "✅ BUY IN ZONE"
    else:
        decision = "👀 WATCH"

    return score, decision, regime

def fetch_nse_market_data():
    print("Querying Institutional Screener for 750 NSE stocks...")
    cols = [
        'name', 'sector', 'close', 'high', 'low', 'change', 'Perf.W', 'Perf.1M',
        'volume', 'average_volume_30d_calc', 'relative_volume_10d_calc',
        'RSI', 'ADX', 'EMA20', 'EMA50', 'ATR', 'beta_1_year',
        'price_52_week_high', 'price_52_week_low',
        'earnings_per_share_forecast_next_fq', 'earnings_per_share_fq',
        'total_revenue_yoy_growth_fq', 'earnings_per_share_diluted_yoy_growth_fq',
        'operating_margin_ttm', 'operating_margin_fq',
        'return_on_equity', 'return_on_equity_fq',
        'earnings_release_next_date', 'price_target_1y',
        'price_earnings_ttm', 'price_book_fq', 'price_book_ratio',
        'debt_to_equity', 'debt_to_equity_fq', 'market_cap_basic'
    ]
    try:
        q = (Query()
             .set_markets('india')
             .select(*cols)
             .where(
                 Column('exchange') == 'NSE',
                 Column('close') >= MIN_PRICE_FILTER,
                 Column('average_volume_30d_calc') > 50000
             )
             .order_by('market_cap_basic', ascending=False)
             .limit(750))
        _, df = q.get_scanner_data()
    except Exception:
        fallback_cols = [
            'name', 'sector', 'close', 'high', 'low', 'change', 'Perf.W', 'Perf.1M',
            'volume', 'average_volume_30d_calc', 'relative_volume_10d_calc',
            'RSI', 'ADX', 'EMA20', 'EMA50', 'ATR', 'beta_1_year',
            'price_52_week_high', 'price_52_week_low',
            'earnings_per_share_forecast_next_fq', 'earnings_per_share_fq',
            'total_revenue_yoy_growth_fq', 'earnings_per_share_diluted_yoy_growth_fq',
            'operating_margin_ttm', 'return_on_equity',
            'earnings_release_next_date',
            'price_earnings_ttm', 'price_book_fq', 'debt_to_equity', 'market_cap_basic'
        ]
        q = (Query()
             .set_markets('india')
             .select(*fallback_cols)
             .where(
                 Column('exchange') == 'NSE',
                 Column('close') >= MIN_PRICE_FILTER,
                 Column('average_volume_30d_calc') > 50000
             )
             .order_by('market_cap_basic', ascending=False)
             .limit(750))
        _, df = q.get_scanner_data()

    print(f"Downloaded {len(df)} NSE stocks successfully.")
    return df

def process_scanner_data(df):
    today = datetime.date.today()
    today_str = str(today)
    nse_catalysts = fetch_nse_corporate_catalysts()
    quant_setups = []
    library_list = []
    price_map = {}
    daily_changes = []

    for _, row in df.iterrows():
        try:
            ticker = str(row.get('name', '')).strip()
            sector = str(row.get('sector') or 'Unknown')
            cmp = round(safe_float(row.get('close'), 0.0), 2)
            if cmp < MIN_PRICE_FILTER:
                continue

            day_high = round(safe_float(row.get('high'), cmp), 2)
            day_low = round(safe_float(row.get('low'), cmp), 2)
            price_map[ticker] = {"cmp": cmp, "high": day_high, "low": day_low}

            chg_1d = round(safe_float(row.get('change'), 0.0), 2)
            daily_changes.append(chg_1d)
            prev_close = round(cmp / (1 + (chg_1d / 100.0)), 2) if chg_1d != -100 else cmp
            chg_1w = round(safe_float(row.get('Perf.W'), 0.0), 2)
            chg_1m = round(safe_float(row.get('Perf.1M'), 0.0), 2)

            vol = safe_float(row.get('volume'), 0.0)
            avg_vol = safe_float(row.get('average_volume_30d_calc'), 1.0)
            rvol_10d = safe_float(row.get('relative_volume_10d_calc'), 0.0)
            vol_mult = round(max(vol / avg_vol if avg_vol > 0 else 0.0, rvol_10d or 0.0), 2)

            rsi = round(safe_float(row.get('RSI'), 0.0), 2)
            adx = round(safe_float(row.get('ADX'), 0.0), 2)
            beta = safe_float(row.get('beta_1_year'))
            ema20 = safe_float(row.get('EMA20'), cmp)
            ema50 = safe_float(row.get('EMA50'), cmp)
            atr = safe_float(row.get('ATR'), cmp * 0.02)

            high_52w = round(safe_float(row.get('price_52_week_high'), cmp), 2)
            low_52w = round(safe_float(row.get('price_52_week_low'), cmp), 2)
            dist_52w = round(((cmp - high_52w) / high_52w) * 100, 2) if high_52w > 0 else 0.0

            qtr_rev_gr = safe_float(row.get('total_revenue_yoy_growth_fq'))
            qtr_eps_gr = safe_float(row.get('earnings_per_share_diluted_yoy_growth_fq'))

            # Consensus Forward EPS Growth with automatic Fallback to Trailing Qtr Profit Growth
            eps_forecast = safe_float(row.get('earnings_per_share_forecast_next_fq'))
            eps_last = safe_float(row.get('earnings_per_share_fq'))
            est_eps_gr = None
            if eps_forecast is not None and eps_last is not None and eps_last > 0:
                est_eps_gr = round(((eps_forecast - eps_last) / abs(eps_last)) * 100, 2)
            eff_eps_gr = est_eps_gr if est_eps_gr is not None else qtr_eps_gr

            op_margin =
