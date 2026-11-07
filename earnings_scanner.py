import os
import datetime
import requests
import pandas as pd
from tradingview_screener import Query, Column

# --- Core Quantitative & Blast Parameters ---
MIN_PRICE_FILTER = 50.0
MIN_RSI = 47.0
MAX_RSI = 78.0
MIN_VOL_SPIKE = 1.25
EMA_PROXIMITY_PCT = 0.06
LIBRARY_DAYS = 30
QUANT_DAYS = 15

def safe_float(val, default=None):
    try:
        if pd.isna(val) or val is None:
            return default
        return float(val)
    except Exception:
        return default

def parse_earnings_date(ts_val, today):
    """Converts Unix timestamp from screener into YYYY-MM-DD and Days Left."""
    if pd.isna(ts_val) or ts_val is None:
        return "Unknown", "N/A"
    try:
        e_date = datetime.datetime.fromtimestamp(int(ts_val)).date()
        days_left = (e_date - today).days
        return str(e_date), days_left
    except Exception:
        return "Unknown", "N/A"

def calculate_blast_score(rsi, vol_mult, dist_52w, chg_1w, chg_1m, qtr_rev_gr, qtr_eps_gr, adx, days_left):
    """
    100-Point Institutional Blast Predictor:
    Combines Technical Ignition (55 pts) + Fundamental Acceleration (35 pts) + Catalyst Proximity (10 pts).
    """
    score = 0
    
    # 1. Technical Momentum & Trend Strength (Max 20 pts)
    if 58 <= rsi <= 72: score += 12
    elif 50 <= rsi < 58: score += 8
    elif 47 <= rsi < 50 or 72 < rsi <= 78: score += 5
    
    if adx is not None and adx >= 25: score += 8
    elif adx is not None and adx >= 18: score += 4

    # 2. Institutional Volume Footprint / RVOL (Max 20 pts)
    if vol_mult >= 2.5: score += 20
    elif vol_mult >= 1.8: score += 15
    elif vol_mult >= 1.3: score += 10
    elif vol_mult >= 1.1: score += 5

    # 3. 52-Week High Breakout Coil & Short-Term RS (Max 15 pts)
    if -5.0 <= dist_52w <= 0: score += 10
    elif -12.0 <= dist_52w < -5.0: score += 7
    elif -20.0 <= dist_52w < -12.0: score += 4

    if (chg_1w or 0) > 2.0 and (chg_1m or 0) > 4.0: score += 5
    elif (chg_1w or 0) > 0: score += 3

    # 4. Quarterly Revenue & Profit Growth Acceleration (Max 35 pts)
    if qtr_rev_gr is not None:
        if qtr_rev_gr >= 25: score += 15
        elif qtr_rev_gr >= 15: score += 11
        elif qtr_rev_gr >= 5: score += 6
    else:
        score += 5 # Neutral weight if unreported

    if qtr_eps_gr is not None:
        if qtr_eps_gr >= 30: score += 20
        elif qtr_eps_gr >= 15: score += 14
        elif qtr_eps_gr > 0: score += 8
    else:
        score += 7 # Neutral weight if unreported

    # 5. Pre-Earnings Catalyst Window (Max 10 pts)
    if isinstance(days_left, int):
        if 1 <= days_left <= 7: score += 10
        elif 8 <= days_left <= 15: score += 7
        elif 16 <= days_left <= 30: score += 4
    else:
        score += 4

    score = min(max(int(round(score)), 0), 100)

    if score >= 85:
        decision = "🚀 BLAST READY"
    elif score >= 72:
        decision = "🔥 STRONG BUY"
    elif score >= 60:
        decision = "✅ BUY"
    else:
        decision = "👀 WATCH"

    return score, decision

def fetch_nse_market_data():
    """Fetches top 750 NSE stocks with full price history, 52W H/L, and Quarterly Growth."""
    print("Querying Institutional Screener for 750 NSE stocks...")
    q = (Query()
         .set_markets('india')
         .select(
             'name', 'sector', 'close', 'change', 'Perf.W', 'Perf.1M',
             'volume', 'average_volume_30d_calc', 'relative_volume_10d_calc',
             'RSI', 'ADX', 'EMA20', 'EMA50', 'ATR',
             'price_52_week_high', 'price_52_week_low',
             'total_revenue_yoy_growth_fq', 'earnings_per_share_diluted_yoy_growth_fq',
             ' earnings_release_next_TradingView_date', 'earnings_release_next_date',
             'price_earnings_ttm', 'price_book_fq', 'debt_to_equity', 'market_cap_basic'
         )
         .where(
             Column('exchange') == 'NSE',
             Column('close') >= MIN_PRICE_FILTER,
             Column('average_volume_30d_calc') > 50000
         )
         .order_by('market_cap_basic', ascending=False)
         .limit(750))

    # Fallback select list if TradingView updates column alias for earnings date
    try:
        _, df = q.get_scanner_data()
    except Exception:
        q = (Query()
             .set_markets('india')
             .select(
                 'name', 'sector', 'close', 'change', 'Perf.W', 'Perf.1M',
                 'volume', 'average_volume_30d_calc', 'relative_volume_10d_calc',
                 'RSI', 'ADX', 'EMA20', 'EMA50', 'ATR',
                 'price_52_week_high', 'price_52_week_low',
                 'total_revenue_yoy_growth_fq', 'earnings_per_share_diluted_yoy_growth_fq',
                 'earnings_release_next_date',
                 'price_earnings_ttm', 'price_book_fq', 'debt_to_equity', 'market_cap_basic'
             )
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
    quant_setups = []
    library_list = []

    for _, row in df.iterrows():
        try:
            ticker = str(row.get('name', '')).strip()
            sector = str(row.get('sector') or 'Unknown')
            cmp = round(safe_float(row.get('close'), 0.0), 2)
            if cmp < MIN_PRICE_FILTER:
                continue

            chg_1d = round(safe_float(row.get('change'), 0.0), 2)
            prev_close = round(cmp / (1 + (chg_1d / 100.0)), 2) if chg_1d != -100 else cmp
            chg_1w = round(safe_float(row.get('Perf.W'), 0.0), 2)
            chg_1m = round(safe_float(row.get('Perf.1M'), 0.0), 2)

            vol = safe_float(row.get('volume'), 0.0)
            avg_vol = safe_float(row.get('average_volume_30d_calc'), 1.0)
            rvol_10d = safe_float(row.get('relative_volume_10d_calc'), 0.0)
            vol_mult = round(max(vol / avg_vol if avg_vol > 0 else 0.0, rvol_10d or 0.0), 2)

            rsi = round(safe_float(row.get('RSI'), 0.0), 2)
            adx = round(safe_float(row.get('ADX'), 0.0), 2)
            ema20 = safe_float(row.get('EMA20'), cmp)
            ema50 = safe_float(row.get('EMA50'), cmp)
            atr = safe_float(row.get('ATR'), cmp * 0.02)

            high_52w = round(safe_float(row.get('price_52_week_high'), cmp), 2)
            low_52w = round(safe_float(row.get('price_52_week_low'), cmp), 2)
            dist_52w = round(((cmp - high_52w) / high_52w) * 100, 2) if high_52w > 0 else 0.0

            qtr_rev_gr = safe_float(row.get('total_revenue_yoy_growth_fq'))
            qtr_eps_gr = safe_float(row.get('earnings_per_share_diluted_yoy_growth_fq'))

            pe = safe_float(row.get('price_earnings_ttm'))
            pb = safe_float(row.get('price_book_fq'))
            debt_eq = safe_float(row.get('debt_to_equity'))
            mcap_cr = round(safe_float(row.get('market_cap_basic'), 0.0) / 10000000, 2)

            e_date_str, days_left = parse_earnings_date(row.get('earnings_release_next_date'), today)

            ai_score, decision = calculate_blast_score(
                rsi, vol_mult, dist_52w, chg_1w, chg_1m, qtr_rev_gr, qtr_eps_gr, adx, days_left
            )

            buy_low = round(min(cmp, ema20), 2)
            buy_zone = f"₹{buy_low} - ₹{cmp}"
            target = round(cmp + max(cmp * 0.07, 2.5 * atr), 2)
            stop_loss = round(cmp - (1.5 * atr), 2)

            record = {
                'Scan Date': today_str,
                'Decision': decision,
                'AI Score': ai_score,
                'Ticker': ticker,
                'Sector': sector,
                'Earnings Date': e_date_str,
                'Days Left': days_left,
                'CMP': cmp,
                'Prev Close': prev_close,
                '1D Chg (%)': chg_1d,
                '1W Chg (%)': chg_1w,
                '1M Chg (%)': chg_1m,
                'Buy Zone': buy_zone,
                'Target': target,
                'Stop Loss': stop_loss,
                'RSI': rsi,
                'Vol Spike': f"{vol_mult:.1f}x",
                'ADX': adx,
                '52W High': high_52w,
                '52W Low': low_52w,
                '52W High Dist (%)': dist_52w,
                'Qtr Sales Gr (%)': round(qtr_rev_gr, 2) if qtr_rev_gr is not None else "N/A",
                'Qtr Profit Gr (%)': round(qtr_eps_gr, 2) if qtr_eps_gr is not None else "N/A",
                'PE Ratio': round(pe, 2) if pe is not None else "N/A",
                'P/B Ratio': round(pb, 2) if pb is not None else "N/A",
                'Debt/Equity': round(debt_eq, 2) if debt_eq is not None else "N/A",
                'MCap (Cr)': mcap_cr
            }

            # 1. Populate 30-Day Earnings Library (Upcoming catalysts or top 300 liquid stocks)
            if (isinstance(days_left, int) and 0 <= days_left <= LIBRARY_DAYS) or len(library_list) < 300:
                library_list.append(record)

            # 2. Strict Filter for "Quant-Fundy Logs" (Blast Setups)
            if not (MIN_RSI <= rsi <= MAX_RSI):
                continue
            if vol_mult < MIN_VOL_SPIKE:
                continue

            near_ema20 = abs(cmp - ema20) / ema20 <= EMA_PROXIMITY_PCT if ema20 > 0 else False
            near_ema50 = abs(cmp - ema50) / ema50 <= EMA_PROXIMITY_PCT if ema50 > 0 else False
            near_52w_breakout = -6.0 <= dist_52w <= 0.5  # Catch tight 52W High consolidations
            if not (near_ema20 or near_ema50 or near_52w_breakout):
                continue

            # Exclude stocks with collapsing quarterly sales/profit if reported
            if qtr_rev_gr is not None and qtr_rev_gr < -10:
                continue
            if qtr_eps_gr is not None and qtr_eps_gr < -15:
                continue

            quant_setups.append(record)

        except Exception:
            continue

    quant_df = pd.DataFrame(quant_setups)
    if not quant_df.empty:
        quant_df = quant_df.sort_values(by=['AI Score', 'RSI'], ascending=[False, False])

    lib_df = pd.DataFrame(library_list)
    if not lib_df.empty:
        lib_df['SortDays'] = pd.to_numeric(lib_df['Days Left'], errors='coerce').fillna(999)
        lib_df = lib_df.sort_values(by=['SortDays', 'AI Score'], ascending=[True, False]).drop(columns=['SortDays'])

    return quant_df, lib_df.to_dict(orient='records')

def push_notifications(quant_df, library_list):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    webhook = os.getenv('WEBHOOK_URL')

    if webhook:
        payload = {
            "reset_headers": True,
            "quant": quant_df.to_dict(orient='records') if not quant_df.empty else [],
            "library": library_list[:300]
        }
        try:
            res = requests.post(webhook, json=payload, headers={'Content-Type': 'application/json'}, timeout=45)
            print(f"Google Sheets Webhook Status: {res.status_code} | Response: {res.text[:200]}")
        except Exception as e:
            print(f"Google Sheets Webhook Failed: {e}")

    if token and chat_id and not quant_df.empty:
        top_df = quant_df[['Decision', 'AI Score', 'Ticker', 'CMP', '1W Chg (%)', 'Target']].head(15)
        msg = (
            f"<b>🚀 QUANT-FUNDY BLAST SCANNER ({len(quant_df)} Setups)</b>\n\n"
            f"<pre>{top_df.to_string(index=False)}</pre>"
        )
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=15
            )
        except Exception as e:
            print(f"Telegram error: {e}")

if __name__ == "__main__":
    raw_df = fetch_nse_market_data()
    quant_df, library_list = process_scanner_data(raw_df)
    print(f"Actionable Blast Setups Found: {len(quant_df)}")
    if not quant_df.empty:
        print(quant_df[['Decision', 'AI Score', 'Ticker', 'CMP', 'Qtr Sales Gr (%)', 'Qtr Profit Gr (%)']].head(10).to_string(index=False))
    push_notifications(quant_df, library_list)
