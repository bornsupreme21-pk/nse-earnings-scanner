import os
import datetime
import requests
import pandas as pd
from tradingview_screener import Query, Column

MIN_PRICE_FILTER = 50.0
MIN_RSI = 46.0
MAX_RSI = 70.0
MAX_1W_CHASE_PCT = 7.5
LIBRARY_MAX_DAYS = 45

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

def evaluate_advance_motion(
    sector, rsi, rvol, chg_1w, chg_1m, est_eps_gr, rev_gr, eps_gr,
    op_m_fq, op_m_ttm, roe, pe, pb, de, days_left, cmp, ema20, ema50, atr
):
    """
    V7.0 Advance Motion Detector:
    Identifies stealth institutional accumulation AT THE LOW BASE before results/events,
    filtering for Margin Expansion + Valuation Headroom so the stock rewards AFTER the result!
    """
    score = 0
    is_fin = any(w in sector.lower() for w in ['finance', 'bank', 'insurance', 'investment'])
    dist_ema20 = ((cmp - ema20) / ema20) * 100 if ema20 > 0 else 0
    atr_pct = (atr / cmp) * 100 if cmp > 0 else 3.0

    # 1. Stealth Motion & Low-Phase Volatility Coil (Max 24 pts)
    if -2.0 <= dist_ema20 <= 2.5 and -2.5 <= chg_1w <= 3.5:
        score += 16  # Tight base coiling right on 20-EMA support
    elif -3.5 <= dist_ema20 <= 4.0 and chg_1w <= 5.5:
        score += 12
    elif dist_ema20 <= 5.5:
        score += 6

    if ema20 >= ema50 * 0.995:
        score += 4  # Structural uptrend alignment (Golden Base)
    if atr_pct <= 3.2:
        score += 4  # Volatility compression (Spring loaded before catalyst)

    # 2. Institutional Volume Absorption & Un-Crowded RSI (Max 18 pts)
    if 48 <= rsi <= 62: score += 9
    elif 62 < rsi <= 68: score += 6
    elif 46 <= rsi < 48: score += 5

    if 1.25 <= rvol <= 2.8: score += 9  # Stealth accumulation without retail blow-off
    elif rvol > 2.8: score += 7
    elif rvol >= 1.05: score += 4

    # 3. Post-Result Reward Engine: Consensus Beat + Margin Expansion (Max 33 pts)
    missing = 0
    if est_eps_gr is not None:
        score += 13 if est_eps_gr >= 18 else (10 if est_eps_gr >= 8 else (5 if est_eps_gr > 0 else 0))
    else:
        score += 10 if (eps_gr or 0) >= 25 else (6 if (eps_gr or 0) >= 12 else 0)

    if rev_gr is not None:
        score += 8 if rev_gr >= 18 else (6 if rev_gr >= 10 else (3 if rev_gr > 0 else 0))
    else: missing += 1

    if eps_gr is not None:
        score += 8 if eps_gr >= 25 else (6 if eps_gr >= 12 else (3 if eps_gr > 0 else 0))
    else: missing += 1

    # Margin Expansion Check (FQ vs TTM): Proves operating leverage is rising into the print
    margin_expanding = (op_m_fq is not None and op_m_ttm is not None and op_m_fq >= op_m_ttm)
    if margin_expanding:
        score += 4

    # 4. Valuation Headroom (Prevents "Priced-In / Sell-the-News" Crashes) (Max 15 pts)
    op_m = op_m_fq if op_m_fq is not None else op_m_ttm
    if is_fin:
        if (roe or 0) >= 15: score += 8
        elif (roe or 0) >= 10: score += 5
        else: missing += 1
        if pb is not None and 0 < pb <= 3.5: score += 7
        elif pe is not None and 0 < pe <= 20: score += 5
    else:
        if (op_m or 0) >= 15: score += 5
        elif (op_m or 0) >= 10: score += 3
        if (roe or 0) >= 16: score += 5
        elif (roe or 0) >= 11: score += 3
        else: missing += 1
        # PEG / Valuation Headroom Check
        if pe is not None and 0 < pe <= 32:
            score += 5
        elif pe is not None and 32 < pe <= 50 and (eps_gr or 0) >= 30:
            score += 4

    # 5. Pre-Result / Post-Result Regime Routing (Max 10 pts)
    val_safe = (pe is None or pe <= 48 or (eps_gr or 0) >= 35)
    regime = "🕵️ STEALTH BASE ACCUMULATION"
    if isinstance(days_left, int):
        if 0 <= days_left <= 1:
            regime = "⛔ BINARY EVENT (Do Not Enter)"
            score -= 25
        elif 2 <= days_left <= 25:
            score += 10
            if val_safe and (margin_expanding or (est_eps_gr or 0) >= 12) and dist_ema20 <= 3.0:
                regime = "🕵️ STEALTH PRE-RESULT HOLD (Reward Post-ER)"
            elif dist_ema20 <= 2.0:
                regime = "💎 LOW-PHASE PRE-ER BUY (Book 50% T-1)"
            else:
                regime = "🎯 PRE-ER RUNUP (Exit T-1)"
        elif 26 <= days_left <= 45:
            score += 8
            regime = "⏳ EARLY PRE-RESULT COIL"
        elif 65 <= days_left <= 92 and (eps_gr or 0) >= 20 and dist_ema20 <= 3.0:
            score += 8
            regime = "⚡ POST-ER PULLBACK CONTINUATION"
        else:
            score += 3
    elif (est_eps_gr or 0) >= 15 and dist_ema20 <= 2.5 and val_safe:
        score += 7
        regime = "🔮 CONSENSUS BEAT STEALTH COIL"

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
            op_m_fq = safe_float(row.get('operating_margin_fq'))
            op_m_ttm = safe_float(row.get('operating_margin_ttm'))
            op_m = op_m_fq if op_m_fq is not None else op_m_ttm
            roe = first_valid(row, ['return_on_equity', 'return_on_equity_fq'])
            pe = safe_float(row.get('price_earnings_ttm'))
            pb = first_valid(row, ['price_book_fq', 'price_book_ratio'])
            de = first_valid(row, ['debt_to_equity', 'debt_to_equity_fq'])
            mcap = round(safe_float(row.get('market_cap_basic'), 0.0) / 10000000, 2)
            target_1y = safe_float(row.get('price_target_1y'))

            e_date_str, days_left = parse_earnings_date(row.get('earnings_release_next_date'), ticker, nse_cat, today)
            ai_score, decision, regime = evaluate_advance_motion(
                sector, rsi, rvol, chg_1w, chg_1m, est_eps_gr, rev_gr, eps_gr,
                op_m_fq, op_m_ttm, roe, pe, pb, de, days_left, cmp, ema20, ema50, atr
            )

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

            # 1. 30-Day Library: Exclude Day 0/1 noise so only actionable 2 to 45 day setups appear!
            if isinstance(days_left, int) and 2 <= days_left <= LIBRARY_MAX_DAYS:
                library_list.append(rec)

            # 2. Strict Filter for "Quant-Fundy Logs" (Stealth Pre-Result & Low-Phase Gems)
            if isinstance(days_left, int) and days_left <= 1: continue
            if not (MIN_RSI <= rsi <= MAX_RSI) or chg_1w > MAX_1W_CHASE_PCT: continue
            near_20 = -3.0 <= ((cmp - ema20) / ema20) * 100 <= 4.0 if ema20 > 0 else False
            near_50 = -2.0 <= ((cmp - ema50) / ema50) * 100 <= 4.0 if ema50 > 0 else False
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
    if not l_df.empty: l_df = l_df.sort_values(by=['AI Score', 'Days Left'], ascending=[False, True])
    return q_df, l_df.to_dict(orient='records'), price_map

def push_notifications(quant_df, library_list, price_map):
    token, chat_id, webhook = os.getenv('TELEGRAM_BOT_TOKEN'), os.getenv('TELEGRAM_CHAT_ID'), os.getenv('WEBHOOK_URL')
    today_str = str(datetime.date.today())

    if token and chat_id and not quant_df.empty:
        try:
            top_df = quant_df[['Ticker', 'Days Left', 'CMP', 'Buy Low', 'Min Target', 'Max Growth (%)']].head(12)
            msg = f"<b>🕵️ V7.0 ADVANCE MOTION PRE-RESULT GEMS ({len(quant_df)})</b>\n<pre>{top_df.to_string(index=False)}</pre>"
            requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=15)
        except Exception as e:
            print(f"Telegram Warning: {e}")

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
            print(f"Sheets Webhook Warning: {e}")

if __name__ == "__main__":
    raw_df = fetch_nse_market_data()
    quant_df, library_list, price_map = process_scanner_data(raw_df)
    print(f"Stealth Gems Found: {len(quant_df)} | Actionable Upcoming Catalysts: {len(library_list)}")
    push_notifications(quant_df, library_list, price_map)
# --- END OF SCRIPT ---
