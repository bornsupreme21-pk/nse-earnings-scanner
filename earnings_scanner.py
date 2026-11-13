import os
import datetime
import requests
import pandas as pd
from tradingview_screener import Query, Column

MIN_PRICE_FILTER = 50.0
MIN_RSI = 45.0
MAX_RSI = 72.0
MAX_1W_CHASE_PCT = 8.5
LIBRARY_MAX_DAYS = 35

NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*"
}

def safe_float(val, default=None):
    try:
        return default if (pd.isna(val) or val is None) else float(val)
    except Exception:
        return default

def first_valid(row, keys, default=None):
    for k in keys:
        v = safe_float(row.get(k))
        if v is not None:
            return v
    return default

def fetch_nse_corporate_catalysts():
    catalysts = {}
    try:
        s = requests.Session()
        s.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=8)
        r = s.get("https://www.nseindia.com/api/corporate-board-meetings?index=equities", headers=NSE_HEADERS, timeout=10)
        if r.status_code == 200:
            for item in r.json():
                sym = str(item.get("bm_symbol", "")).strip()
                purp = str(item.get("bm_purpose", "")).strip()
                d_str = str(item.get("bm_date", "")).strip()
                if sym and d_str:
                    dt = datetime.datetime.strptime(d_str, "%d-%b-%Y").date()
                    tag = "📊 Qtr Results BM" if "Result" in purp else "📅 Board Meeting"
                    if any(k in purp.lower() for k in ["acqui", "merger", "scheme"]): tag = "🤝 M&A / Restructuring"
                    elif any(k in purp.lower() for k in ["dividend", "buyback", "bonus"]): tag = "🎁 Corp Action BM"
                    catalysts[sym] = (dt, tag)
    except Exception:
        pass
    return catalysts

def parse_earnings_date(ts_val, ticker, nse_cat, today):
    if ticker in nse_cat:
        dt, tag = nse_cat[ticker]
        days = (dt - today).days
        if days >= 0:
            return f"{dt} ({tag})", int(days)
    if pd.isna(ts_val) or ts_val is None:
        return "Unknown", ""
    try:
        e_date = datetime.datetime.fromtimestamp(int(ts_val)).date()
        return str(e_date), int((e_date - today).days)
    except Exception:
        return "Unknown", ""

def evaluate_low_phase(sector, rsi, rvol, chg_1w, est_eps_gr, rev_gr, eps_gr, op_m, roe, pe, pb, de, days_left, cmp, ema20):
    score = 0
    is_fin = any(w in sector.lower() for w in ['finance', 'bank', 'insurance', 'investment'])
    dist_ema20 = ((cmp - ema20) / ema20) * 100 if ema20 > 0 else 0

    if -2.5 <= dist_ema20 <= 2.5 and -3.0 <= chg_1w <= 4.0: score += 22
    elif -4.0 <= dist_ema20 <= 4.5 and chg_1w <= 6.5: score += 16
    elif dist_ema20 <= 6.0: score += 9

    if 48 <= rsi <= 63: score += 10
    elif 63 < rsi <= 70: score += 6
    elif 45 <= rsi < 48: score += 5

    if rvol >= 2.0: score += 10
    elif rvol >= 1.4: score += 8
    elif rvol >= 1.1: score += 5

    missing = 0
    if est_eps_gr is not None:
        score += 13 if est_eps_gr >= 20 else (10 if est_eps_gr >= 10 else (6 if est_eps_gr > 0 else 0))
    else:
        score += 10 if (eps_gr or 0) >= 25 else (6 if (eps_gr or 0) >= 12 else 0)

    if rev_gr is not None:
        score += 10 if rev_gr >= 20 else (7 if rev_gr >= 10 else (4 if rev_gr > 0 else 0))
    else: missing += 1

    if eps_gr is not None:
        score += 10 if eps_gr >= 25 else (7 if eps_gr >= 12 else (4 if eps_gr > 0 else 0))
    else: missing += 1

    if is_fin:
        if (roe or 0) >= 15: score += 9
        elif (roe or 0) >= 10: score += 5
        else: missing += 1
        if pb is not None and 0 < pb <= 4.0: score += 6
        elif pe is not None and 0 < pe <= 22: score += 4
    else:
        if (op_m or 0) >= 16: score += 6
        elif (op_m or 0) >= 10: score += 4
        if (roe or 0) >= 16: score += 6
        elif (roe or 0) >= 10: score += 4
        else: missing += 1
        if de is not None and de <= 0.6: score += 3

    regime = "💎 LOW-PHASE BASE BUY"
    if isinstance(days_left, int):
        if 0 <= days_left <= 1:
            regime = "⛔ BINARY EVENT (Do Not Enter)"
            score -= 20
        elif 2 <= days_left <= 15:
            score += 10
            regime = "💎 PRE-EVENT LOW-PHASE BUY (Exit T-1)" if dist_ema20 <= 2.0 else "🎯 PRE-ER ANTICIPATION (Exit T-1)"
        elif 16 <= days_left <= 35:
            score += 8
            regime = "⏳ EARLY PRE-RESULT ACCUMULATION"
        elif 65 <= days_left <= 92 and (eps_gr or 0) >= 20 and dist_ema20 <= 3.5:
            score += 7
            regime = "⚡ POST-RESULT PULLBACK BUY"
        else: score += 3
    elif (est_eps_gr or 0) >= 15 and dist_ema20 <= 2.5:
        score += 6
        regime = "🔮 ESTIMATE BEAT ACCUMULATION"

    if missing >= 2: score = min(score, 73)
    score = min(max(int(round(score)), 0), 100)

    if "BINARY EVENT" in regime: decision = "⛔ AVOID EVENT GAP"
    elif score >= 84: decision = "🚀 BLAST READY (Buy Low)"
    elif score >= 74: decision = "🔥 STRONG BUY"
    elif score >= 64: decision = "✅ ACCUMULATE"
    else: decision = "👀 WATCH"
    return score, decision, regime

def fetch_nse_market_data():
    base_cols = [
        'name', 'sector', 'close', 'high', 'low', 'change', 'Perf.W', 'Perf.1M',
        'volume', 'average_volume_30d_calc', 'relative_volume_10d_calc',
        'RSI', 'ADX', 'EMA20', 'EMA50', 'ATR', 'beta_1_year',
        'price_52_week_high', 'price_52_week_low',
        'earnings_per_share_forecast_next_fq', 'earnings_per_share_fq',
        'total_revenue_yoy_growth_fq', 'earnings_per_share_diluted_yoy_growth_fq',
        'operating_margin_ttm', 'return_on_equity', 'earnings_release_next_date',
        'price_earnings_ttm', 'price_book_fq', 'debt_to_equity', 'market_cap_basic'
    ]
    ext_cols = base_cols + ['operating_margin_fq', 'return_on_equity_fq', 'price_target_1y', 'price_book_ratio', 'debt_to_equity_fq']
    for col_set in [ext_cols, base_cols]:
        try:
            q = (Query().set_markets('india').select(*col_set)
                 .where(Column('exchange') == 'NSE', Column('close') >= MIN_PRICE_FILTER, Column('average_volume_30d_calc') > 50000)
                 .order_by('market_cap_basic', ascending=False).limit(750))
            _, df = q.get_scanner_data()
            return df
        except Exception:
            continue
    return pd.DataFrame()

def process_scanner_data(df):
    today = datetime.date.today()
    today_str = str(today)
    nse_cat = fetch_nse_corporate_catalysts()
    quant_setups, library_list, price_map = [], [], {}

    for _, row in df.iterrows():
        try:
            ticker = str(row.get('name', '')).strip()
            sector = str(row.get('sector') or 'Unknown')
            cmp = round(safe_float(row.get('close'), 0.0), 2)
            if cmp < MIN_PRICE_FILTER: continue

            price_map[ticker] = {"cmp": cmp, "high": round(safe_float(row.get('high'), cmp), 2), "low": round(safe_float(row.get('low'), cmp), 2)}
            chg_1d = round(safe_float(row.get('change'), 0.0), 2)
            prev_close = round(cmp / (1 + (chg_1d / 100.0)), 2) if chg_1d != -100 else cmp
            chg_1w = round(safe_float(row.get('Perf.W'), 0.0), 2)
            chg_1m = round(safe_float(row.get('Perf.1M'), 0.0), 2)

            vol = safe_float(row.get('volume'), 0.0)
            avg_vol = safe_float(row.get('average_volume_30d_calc'), 1.0)
            rvol_10d = safe_float(row.get('relative_volume_10d_calc'), 0.0)
            rvol = round(max(vol / avg_vol if avg_vol > 0 else 0.0, rvol_10d or 0.0), 2)

            rsi = round(safe_float(row.get('RSI'), 0.0), 2)
            adx = round(safe_float(row.get('ADX'), 0.0), 2)
            beta = safe_float(row.get('beta_1_year'))
            ema20 = safe_float(row.get('EMA20'), cmp)
            ema50 = safe_float(row.get('EMA50'), cmp)
            atr = safe_float(row.get('ATR'), cmp * 0.02)

            h52 = round(safe_float(row.get('price_52_week_high'), cmp), 2)
            l52 = round(safe_float(row.get('price_52_week_low'), cmp), 2)
            dist_52w = round(((cmp - h52) / h52) * 100, 2) if h52 > 0 else 0.0

            eps_f = safe_float(row.get('earnings_per_share_forecast_next_fq'))
            eps_l = safe_float(row.get('earnings_per_share_fq'))
            est_eps_gr = round(((eps_f - eps_l) / abs(eps_l)) * 100, 2) if (eps_f is not None and eps_l is not None and eps_l > 0) else None

            rev_gr = safe_float(row.get('total_revenue_yoy_growth_fq'))
            eps_gr = safe_float(row.get('earnings_per_share_diluted_yoy_growth_fq'))
            op_m = first_valid(row, ['operating_margin_ttm', 'operating_margin_fq'])
            roe = first_valid(row, ['return_on_equity', 'return_on_equity_fq'])
            pe = safe_float(row.get('price_earnings_ttm'))
            pb = first_valid(row, ['price_book_fq', 'price_book_ratio'])
            de = first_valid(row, ['debt_to_equity', 'debt_to_equity_fq'])
            mcap = round(safe_float(row.get('market_cap_basic'), 0.0) / 10000000, 2)
            target_1y = safe_float(row.get('price_target_1y'))

            e_date_str, days_left = parse_earnings_date(row.get('earnings_release_next_date'), ticker, nse_cat, today)
            ai_score, decision, regime = evaluate_low_phase(sector, rsi, rvol, chg_1w, est_eps_gr, rev_gr, eps_gr, op_m, roe, pe, pb, de, days_left, cmp, ema20)

            anchor = min(cmp, max(ema20, cmp - 0.8 * atr))
            buy_low = round(anchor - (0.4 * atr), 2)
            buy_high = round(min(cmp, ema20 * 1.015), 2) if cmp > ema20 * 1.015 else cmp
            min_target = round(cmp * 1.0525, 2)
            tech_max = round(cmp + max(cmp * 0.09, 3.0 * atr), 2)
            max_target = max(round(min(target_1y, cmp * 1.22), 2), tech_max) if (target_1y and target_1y > min_target) else tech_max
            max_gr_pct = round(((max_target - cmp) / cmp) * 100, 2)
            stop_loss = round(min(buy_low - 0.8 * atr, cmp * 0.965), 2)

            rec = {
                'Scan Date': today_str, 'Regime': regime, 'Decision': decision, 'AI Score': ai_score,
                'Ticker': ticker, 'Sector': sector, 'Earnings Date': e_date_str, 'Days Left': days_left,
                'CMP': cmp, 'Prev Close': prev_close, '1D Chg (%)': chg_1d, '1W Chg (%)': chg_1w, '1M Chg (%)': chg_1m,
                'Buy Low': buy_low, 'Buy High': buy_high, 'Min Target': min_target, 'Max Target': max_target,
                'Max Growth (%)': max_gr_pct, 'Stop Loss': stop_loss, 'RSI': rsi, 'RVOL': rvol, 'ADX': adx,
                '1Y Beta': round(beta, 2) if beta is not None else "", '52W High': h52, '52W Low': l52, '52W High Dist (%)': dist_52w,
                'Est Next EPS Gr (%)': est_eps_gr if est_eps_gr is not None else "",
                'Qtr Sales Gr (%)': round(rev_gr, 2) if rev_gr is not None else "",
                'Qtr Profit Gr (%)': round(eps_gr, 2) if eps_gr is not None else "",
                'Op Margin (%)': round(op_m, 2) if op_m is not None else "",
                'ROE (%)': round(roe, 2) if roe is not None else "",
                'PE Ratio': round(pe, 2) if pe is not None else "",
                'P/B Ratio': round(pb, 2) if pb is not None else "",
                'Debt/Equity': round(de, 2) if de is not None else "", 'MCap (Cr)': mcap
            }

            if isinstance(days_left, int) and 1 <= days_left <= LIBRARY_MAX_DAYS:
                library_list.append(rec)

            if isinstance(days_left, int) and days_left <= 1: continue
            if not (MIN_RSI <= rsi <= MAX_RSI) or chg_1w > MAX_1W_CHASE_PCT: continue
            near_20 = -3.0 <= ((cmp - ema20) / ema20) * 100 <= 4.5 if ema20 > 0 else False
            near_50 = -2.0 <= ((cmp - ema50) / ema50) * 100 <= 4.5 if ema50 > 0 else False
            if not (near_20 or near_50): continue

            is_fin = any(w in sector.lower() for w in ['finance', 'bank', 'insurance', 'investment'])
            if (rev_gr is not None and rev_gr < 0) or (eps_gr is not None and eps_gr < 0): continue
            if not is_fin and op_m is not None and op_m <= 5.0: continue
            if ai_score >= 74: quant_setups.append(rec)
        except Exception:
            continue

    q_df = pd.DataFrame(quant_setups)
    if not q_df.empty: q_df = q_df.sort_values(by=['AI Score', 'Max Growth (%)'], ascending=[False, False])
    l_df = pd.DataFrame(library_list)
    if not l_df.empty: l_df = l_df.sort_values(by=['Days Left', 'AI Score'], ascending=[True, False])
    return q_df, l_df.to_dict(orient='records'), price_map

def push_notifications(quant_df, library_list, price_map):
    token, chat_id, webhook = os.getenv('TELEGRAM_BOT_TOKEN'), os.getenv('TELEGRAM_CHAT_ID'), os.getenv('WEBHOOK_URL')
    today_str = str(datetime.date.today())

    # 1. Send Telegram Alert First
    if token and chat_id and not quant_df.empty:
        try:
            top_df = quant_df[['Ticker', 'Days Left', 'CMP', 'Buy Low', 'Min Target', 'Max Growth (%)']].head(12)
            msg = f"<b>💎 V6.0 LOW-PHASE PRE-EVENT GEMS ({len(quant_df)})</b>\n<pre>{top_df.to_string(index=False)}</pre>"
            tg_res = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=15)
            print(f"Telegram Status: {tg_res.status_code}")
        except Exception as e:
            print(f"Telegram Warning: {e}")

    # 2. Push to Google Sheets with 90s Timeout & Safe Exception Handling
    if webhook:
        try:
            payload = {
                "reset_logs": False,
                "scan_date": today_str,
                "quant": quant_df.to_dict(orient='records') if not quant_df.empty else [],
                "library": library_list,
                "price_map": price_map
            }
            res = requests.post(webhook, json=payload, headers={'Content-Type': 'application/json'}, timeout=90)
            print(f"Sheets Webhook: {res.status_code} | {res.text[:150]}")
        except Exception as e:
            print(f"Sheets Webhook Warning (Data still processed by Google in background): {e}")

if __name__ == "__main__":
    raw_df = fetch_nse_market_data()
    quant_df, library_list, price_map = process_scanner_data(raw_df)
    print(f"Gems Found: {len(quant_df)} | 30-Day Catalysts: {len(library_list)}")
    if not quant_df.empty:
        print(quant_df[['Regime', 'AI Score', 'Ticker', 'Days Left', 'CMP', 'Buy Low', 'Min Target', 'Max Growth (%)']].head(15).to_string(index=False))
    push_notifications(quant_df, library_list, price_map)
# --- END OF SCRIPT ---
