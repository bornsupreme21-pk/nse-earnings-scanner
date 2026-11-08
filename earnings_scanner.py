import os
import datetime
import requests
import pandas as pd
from tradingview_screener import Query, Column

# --- Core Quantitative & Blast Parameters ---
MIN_PRICE_FILTER = 50.0
MIN_RSI = 48.0
MAX_RSI = 78.0
MIN_VOL_SPIKE = 1.3
EMA_PROXIMITY_PCT = 0.06
LIBRARY_MAX_DAYS = 35

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

def parse_earnings_date(ts_val, today):
    if pd.isna(ts_val) or ts_val is None:
        return "Unknown", ""
    try:
        e_date = datetime.datetime.fromtimestamp(int(ts_val)).date()
        days_left = (e_date - today).days
        return str(e_date), int(days_left)
    except Exception:
        return "Unknown", ""

def classify_regime_and_score(
    sector, rsi, vol_mult, dist_52w, chg_1d, chg_1w, chg_1m,
    qtr_rev_gr, qtr_eps_gr, op_margin, roe, pe, pb, debt_eq, adx, days_left, cmp, ema20
):
    """
    Sector-Aware, 100-Point Institutional Scoring Engine with Missing-Data Penalty
    and Counter-Strategy Regime Classification.
    """
    score = 0
    is_financial = any(w in sector.lower() for w in ['finance', 'bank', 'insurance', 'investment'])

    # 1. Technical Momentum & Trend Strength (Max 15 pts)
    if 58 <= rsi <= 72: score += 10
    elif 50 <= rsi < 58: score += 7
    elif 48 <= rsi < 50 or 72 < rsi <= 78: score += 4

    if adx is not None and adx >= 25: score += 5
    elif adx is not None and adx >= 18: score += 3

    # 2. Institutional Volume Footprint / RVOL (Max 18 pts)
    if vol_mult >= 3.0: score += 18
    elif vol_mult >= 2.0: score += 14
    elif vol_mult >= 1.5: score += 10
    elif vol_mult >= 1.25: score += 6

    # 3. 52-Week High Proximity & Multi-Timeframe RS (Max 17 pts)
    if -4.0 <= dist_52w <= 0: score += 11
    elif -10.0 <= dist_52w < -4.0: score += 8
    elif -18.0 <= dist_52w < -10.0: score += 4

    if (chg_1w or 0) >= 4.0 and (chg_1m or 0) >= 6.0: score += 6
    elif (chg_1w or 0) > 0 and (chg_1m or 0) > 0: score += 3

    # 4. Quarterly Revenue & Profit Acceleration (Max 30 pts)
    missing_fundy_count = 0
    if qtr_rev_gr is not None:
        if qtr_rev_gr >= 25: score += 13
        elif qtr_rev_gr >= 15: score += 10
        elif qtr_rev_gr >= 8: score += 6
    else:
        missing_fundy_count += 1

    if qtr_eps_gr is not None:
        if qtr_eps_gr >= 35: score += 17
        elif qtr_eps_gr >= 20: score += 13
        elif qtr_eps_gr >= 10: score += 8
    else:
        missing_fundy_count += 1

    # 5. Sector-Aware Quality & Balance Sheet Health (Max 15 pts)
    if is_financial:
        # Banks/NBFCs: Score on ROE and P/B valuation rather than Op Margin & Debt/Eq
        if roe is not None:
            if roe >= 18: score += 10
            elif roe >= 13: score += 7
            elif roe >= 8: score += 4
        else:
            missing_fundy_count += 1
        if pb is not None and 0 < pb <= 4.5:
            score += 5
        elif pe is not None and 0 < pe <= 25:
            score += 4
    else:
        # Non-Financials: Score on Operating Margin, ROE, and Low Leverage
        if op_margin is not None:
            if op_margin >= 20: score += 6
            elif op_margin >= 12: score += 4
            elif op_margin >= 6: score += 2
        else:
            missing_fundy_count += 1

        if roe is not None:
            if roe >= 20: score += 6
            elif roe >= 14: score += 4
            elif roe >= 10: score += 2
        else:
            missing_fundy_count += 1

        if debt_eq is not None:
            if debt_eq <= 0.5: score += 3
            elif debt_eq <= 1.0: score += 1
        else:
            missing_fundy_count += 1

    # 6. Catalyst Timing & Regime Classification (Max 5 pts)
    regime = "📈 MOMENTUM SWING"
    if isinstance(days_left, int):
        if 0 <= days_left <= 1:
            regime = "⛔ BINARY EVENT (Hold/No New Buy)"
            score -= 15  # Never recommend fresh entry on Day 0/1
        elif 2 <= days_left <= 15:
            score += 5
            if chg_1d < 0 and cmp <= ema20 * 1.02:
                regime = "🪤 PRE-ER PULLBACK BUY (Exit T-1)"
            else:
                regime = "🎯 PRE-ER RUNUP (Exit T-1)"
        elif 16 <= days_left <= 35:
            score += 4
            regime = "⏳ EARLY PRE-ER ACCUMULATION"
        elif 65 <= days_left <= 95 and (qtr_eps_gr or 0) >= 25 and vol_mult >= 1.5:
            score += 5
            regime = "⚡ POST-ER DRIFT (PEAD Breakout)"
        else:
            score += 2

    # Reliability Penalty: Prevent incomplete N/A rows from hitting 90+
    if missing_fundy_count >= 2:
        score = min(score, 74)

    score = min(max(int(round(score)), 0), 100)

    if "BINARY EVENT" in regime:
        decision = "⛔ AVOID EVENT GAP"
    elif score >= 85:
        decision = "🚀 BLAST READY"
    elif score >= 75:
        decision = "🔥 STRONG BUY"
    elif score >= 62:
        decision = "✅ BUY"
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
        'total_revenue_yoy_growth_fq', 'earnings_per_share_diluted_yoy_growth_fq',
        'operating_margin_ttm', 'operating_margin_fq',
        'return_on_equity', 'return_on_equity_fq',
        'earnings_release_next_date',
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
        # Fallback if a secondary column alias isn't supported
        fallback_cols = [
            'name', 'sector', 'close', 'high', 'low', 'change', 'Perf.W', 'Perf.1M',
            'volume', 'average_volume_30d_calc', 'relative_volume_10d_calc',
            'RSI', 'ADX', 'EMA20', 'EMA50', 'ATR', 'beta_1_year',
            'price_52_week_high', 'price_52_week_low',
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
    quant_setups = []
    library_list = []
    price_map = {}

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
            op_margin = first_valid_float(row, ['operating_margin_ttm', 'operating_margin_fq'])
            roe = first_valid_float(row, ['return_on_equity', 'return_on_equity_fq'])

            pe = safe_float(row.get('price_earnings_ttm'))
            pb = first_valid_float(row, ['price_book_fq', 'price_book_ratio'])
            debt_eq = first_valid_float(row, ['debt_to_equity', 'debt_to_equity_fq'])
            mcap_cr = round(safe_float(row.get('market_cap_basic'), 0.0) / 10000000, 2)

            e_date_str, days_left = parse_earnings_date(row.get('earnings_release_next_date'), today)

            ai_score, decision, regime = classify_regime_and_score(
                sector, rsi, vol_mult, dist_52w, chg_1d, chg_1w, chg_1m,
                qtr_rev_gr, qtr_eps_gr, op_margin, roe, pe, pb, debt_eq, adx, days_left, cmp, ema20
            )

            # Structural ATR & EMA Support Brackets (Pure Numbers)
            buy_low = round(min(cmp, max(ema20, cmp - 1.0 * atr)), 2)
            buy_high = cmp
            target = round(cmp + max(cmp * 0.07, 2.5 * atr), 2)
            stop_loss = round(min(ema20 - 0.5 * atr, cmp - 1.5 * atr), 2)

            record = {
                'Scan Date': today_str,
                'Regime': regime,
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
                'Buy Low': buy_low,
                'Buy High': buy_high,
                'Target': target,
                'Stop Loss': stop_loss,
                'RSI': rsi,
                'RVOL': vol_mult,
                'ADX': adx,
                '1Y Beta': round(beta, 2) if beta is not None else "",
                '52W High': high_52w,
                '52W Low': low_52w,
                '52W High Dist (%)': dist_52w,
                'Qtr Sales Gr (%)': round(qtr_rev_gr, 2) if qtr_rev_gr is not None else "",
                'Qtr Profit Gr (%)': round(qtr_eps_gr, 2) if qtr_eps_gr is not None else "",
                'Op Margin (%)': round(op_margin, 2) if op_margin is not None else "",
                'ROE (%)': round(roe, 2) if roe is not None else "",
                'PE Ratio': round(pe, 2) if pe is not None else "",
                'P/B Ratio': round(pb, 2) if pb is not None else "",
                'Debt/Equity': round(debt_eq, 2) if debt_eq is not None else "",
                'MCap (Cr)': mcap_cr
            }

            # 1. STRICT 30-Day Earnings Library Filter: Only genuine upcoming catalysts (1 to 35 days out)
            if isinstance(days_left, int) and 1 <= days_left <= LIBRARY_MAX_DAYS:
                library_list.append(record)

            # 2. Strict Filter for "Quant-Fundy Logs" (Actionable Setups Only)
            if isinstance(days_left, int) and days_left <= 1:
                continue  # Block Day-0 and Day-1 binary event gambles
            if not (MIN_RSI <= rsi <= MAX_RSI):
                continue
            if vol_mult < MIN_VOL_SPIKE:
                continue

            near_ema20 = abs(cmp - ema20) / ema20 <= EMA_PROXIMITY_PCT if ema20 > 0 else False
            near_ema50 = abs(cmp - ema50) / ema50 <= EMA_PROXIMITY_PCT if ema50 > 0 else False
            near_52w_breakout = -5.5 <= dist_52w <= 0.5
            if not (near_ema20 or near_ema50 or near_52w_breakout):
                continue

            is_fin = any(w in sector.lower() for w in ['finance', 'bank', 'insurance', 'investment'])
            if qtr_rev_gr is not None and qtr_rev_gr < 0:
                continue
            if qtr_eps_gr is not None and qtr_eps_gr < 0:
                continue
            if not is_fin and op_margin is not None and op_margin <= 4.0:
                continue

            if ai_score >= 72:
                quant_setups.append(record)

        except Exception:
            continue

    quant_df = pd.DataFrame(quant_setups)
    if not quant_df.empty:
        quant_df = quant_df.sort_values(by=['AI Score', 'RVOL'], ascending=[False, False])

    lib_df = pd.DataFrame(library_list)
    if not lib_df.empty:
        lib_df = lib_df.sort_values(by=['Days Left', 'AI Score'], ascending=[True, False])

    return quant_df, lib_df.to_dict(orient='records'), price_map

def push_notifications(quant_df, library_list, price_map):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    webhook = os.getenv('WEBHOOK_URL')
    today_str = str(datetime.date.today())

    if webhook:
        payload = {
            "reset_logs": False,
            "scan_date": today_str,
            "quant": quant_df.to_dict(orient='records') if not quant_df.empty else [],
            "library": library_list,
            "price_map": price_map
        }
        try:
            res = requests.post(webhook, json=payload, headers={'Content-Type': 'application/json'}, timeout=45)
            print(f"Google Sheets Webhook Status: {res.status_code} | Response: {res.text[:200]}")
        except Exception as e:
            print(f"Google Sheets Webhook Failed: {e}")

    if token and chat_id and not quant_df.empty:
        top_df = quant_df[['Decision', 'AI Score', 'Ticker', 'Days Left', 'CMP', 'Target']].head(15)
        msg = (
            f"<b>🚀 V5.0 QUANT-FUNDY BLAST ENGINE ({len(quant_df)} Elite Setups)</b>\n\n"
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
    quant_df, library_list, price_map = process_scanner_data(raw_df)
    print(f"Elite Setups Found: {len(quant_df)} | True 30-Day Catalysts: {len(library_list)}")
    if not quant_df.empty:
        print(quant_df[['Regime', 'AI Score', 'Ticker', 'Days Left', 'CMP', 'RVOL']].head(12).to_string(index=False))
    push_notifications(quant_df, library_list, price_map)
