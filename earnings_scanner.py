import os
import datetime
import requests
import pandas as pd
from tradingview_screener import Query, Column

# --- Core Low-Phase Pre-Catalyst Parameters ---
MIN_PRICE_FILTER = 50.0
MIN_RSI = 45.0
MAX_RSI = 72.0           # Lowered from 78 so we NEVER buy overbought tops!
MAX_1W_CHASE_PCT = 8.5   # Reject stocks that have already spiked >8.5% this week
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
    """Fetches upcoming NSE Board Meetings (Results, Acquisitions, Fundraising, Dividends) if reachable."""
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
                        elif any(k in purpose.lower() for k in ["acqui", "amalga", "merger", "scheme"]): tag = "🤝 M&A / Restructuring"
                        elif any(k in purpose.lower() for k in ["fund", "preferential", "qip", "rights"]): tag = "💰 Capital Raise BM"
                        elif any(k in purpose.lower() for k in ["dividend", "buyback", "bonus", "split"]): tag = "🎁 Corporate Action BM"
                        catalysts[sym] = (dt, tag)
                    except Exception:
                        pass
    except Exception:
        pass
    return catalysts

def parse_earnings_date(ts_val, ticker, nse_catalysts, today):
    # Check NSE Official Board Meeting first
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

def evaluate_low_phase_setup(
    sector, rsi, vol_mult, dist_52w, chg_1d, chg_1w, chg_1m,
    est_eps_gr, qtr_rev_gr, qtr_eps_gr, op_margin, roe, pe, pb, debt_eq,
    adx, days_left, cmp, ema20, ema50, atr
):
    """
    Scores stocks specifically for LOW-PHASE ACCUMULATION before an upcoming event/result
    where 5%+ minimum upside is achievable without chasing overextended highs.
    """
    score = 0
    is_financial = any(w in sector.lower() for w in ['finance', 'bank', 'insurance', 'investment'])

    # 1. Low-Phase Base Proximity (Max 22 pts) - Rewards buying near support, NOT after a huge rally!
    dist_ema20_pct = ((cmp - ema20) / ema20) * 100 if ema20 > 0 else 0
    if -2.5 <= dist_ema20_pct <= 2.5 and -3.0 <= chg_1w <= 4.0:
        score += 22  # Perfect quiet coiling at 20-EMA support base
    elif -4.0 <= dist_ema20_pct <= 4.5 and chg_1w <= 6.5:
        score += 16  # Healthy accumulation zone
    elif dist_ema20_pct <= 6.0:
        score += 9

    # 2. Optimal Pre-Breakout RSI & Institutional Volume Absorption (Max 20 pts)
    if 48 <= rsi <= 63:
        score += 10  # Plenty of room to run before hitting overbought 70+
    elif 63 < rsi <= 70:
        score += 6
    elif 45 <= rsi < 48:
        score += 5

    if vol_mult >= 2.0: score += 10
    elif vol_mult >= 1.4: score += 8
    elif vol_mult >= 1.1: score += 5

    # 3. Forward Earnings Forecast & Recent Fundamental Acceleration (Max 33 pts)
    missing_fundy = 0
    if est_eps_gr is not None:
        if est_eps_gr >= 20: score += 13
        elif est_eps_gr >= 10: score += 10
        elif est_eps_gr > 0: score += 6
    else:
        # Use historical momentum if analyst consensus isn't covered
        if (qtr_eps_gr or 0) >= 25: score += 10
        elif (qtr_eps_gr or 0) >= 12: score += 6

    if qtr_rev_gr is not None:
        if qtr_rev_gr >= 20: score += 10
        elif qtr_rev_gr >= 10: score += 7
        elif qtr_rev_gr > 0: score += 4
    else:
        missing_fundy += 1

    if qtr_eps_gr is not None:
        if qtr_eps_gr >= 25: score += 10
        elif qtr_eps_gr >= 12: score += 7
        elif qtr_eps_gr > 0: score += 4
    else:
        missing_fundy += 1

    # 4. Sector-Aware Quality & Valuation Safety (Max 15 pts)
    if is_financial:
        if (roe or 0) >= 15: score += 9
        elif (roe or 0) >= 10: score += 5
        else: missing_fundy += 1
        if pb is not None and 0 < pb <= 4.0: score += 6
        elif pe is not None and 0 < pe <= 22: score += 4
    else:
        if (op_margin or 0) >= 16: score += 6
        elif (op_margin or 0) >= 10: score += 4
        if (roe or 0) >= 16: score += 6
        elif (roe or 0) >= 10: score += 4
        else: missing_fundy += 1
        if debt_eq is not None and debt_eq <= 0.6: score += 3

    # 5. Event / Earnings Catalyst Horizon (Max 10 pts)
    regime = "💎 LOW-PHASE BASE BUY"
    if isinstance(days_left, int):
        if 0 <= days_left <= 1:
            regime = "⛔ BINARY EVENT (Do Not Enter)"
            score -= 20
        elif 2 <= days_left <= 15:
            score += 10
            if dist_ema20_pct <= 2.0:
                regime = "💎 PRE-EVENT LOW-PHASE BUY (Exit T-1)"
            else:
                regime = "🎯 PRE-ER ANTICIPATION (Exit T-1)"
        elif 16 <= days_left <= 35:
            score += 8
            regime = "⏳ EARLY PRE-RESULT ACCUMULATION"
        elif 65 <= days_left <= 92 and (qtr_eps_gr or 0) >= 20 and dist_ema20_pct <= 3.5:
            score += 7
            regime = "⚡ POST-RESULT PULLBACK BUY"
        else:
            score += 3
    else:
        if (est_eps_gr or 0) >= 15 and dist_ema20_pct <= 2.5:
            score += 6
            regime = "🔮 ESTIMATE BEAT ACCUMULATION"

    if missing_fundy >= 2:
        score = min(score, 73)

    score = min(max(int(round(score)), 0), 100)

    if "BINARY EVENT" in regime:
        decision = "⛔ AVOID EVENT GAP"
    elif score >= 84:
        decision = "🚀 BLAST READY (Buy Low)"
    elif score >= 74:
        decision = "🔥 STRONG BUY"
    elif score >= 64:
        decision = "✅ ACCUMULATE"
    else:
        decision = "👀 WATCH"

    return score, decision, regime

def fetch_nse_market_data():
    print("Querying Institutional Screener + Consensus Estimates for 750 NSE stocks...")
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

            # Forward Consensus EPS Surprise Calculation
            eps_forecast = safe_float(row.get('earnings_per_share_forecast_next_fq'))
            eps_last = safe_float(row.get('earnings_per_share_fq'))
            est_eps_gr = None
            if eps_forecast is not None and eps_last is not None and eps_last > 0:
                est_eps_gr = round(((eps_forecast - eps_last) / abs(eps_last)) * 100, 2)

            qtr_rev_gr = safe_float(row.get('total_revenue_yoy_growth_fq'))
            qtr_eps_gr = safe_float(row.get('earnings_per_share_diluted_yoy_growth_fq'))
            op_margin = first_valid_float(row, ['operating_margin_ttm', 'operating_margin_fq'])
            roe = first_valid_float(row, ['return_on_equity', 'return_on_equity_fq'])

            pe = safe_float(row.get('price_earnings_ttm'))
            pb = first_valid_float(row, ['price_book_fq', 'price_book_ratio'])
            debt_eq = first_valid_float(row, ['debt_to_equity', 'debt_to_equity_fq'])
            mcap_cr = round(safe_float(row.get('market_cap_basic'), 0.0) / 10000000, 2)
            analyst_target = safe_float(row.get('price_target_1y'))

            e_date_str, days_left = parse_earnings_date(row.get('earnings_release_next_date'), ticker, nse_catalysts, today)

            ai_score, decision, regime = evaluate_low_phase_setup(
                sector, rsi, vol_mult, dist_52w, chg_1d, chg_1w, chg_1m,
                est_eps_gr, qtr_rev_gr, qtr_eps_gr, op_margin, roe, pe, pb, debt_eq,
                adx, days_left, cmp, ema20, ema50, atr
            )

            # Buying Range at the Low Phase & Dual Target Brackets (Min 5.25% + Max Catalyst Upside)
            support_anchor = min(cmp, max(ema20, cmp - 0.8 * atr))
            buy_low = round(support_anchor - (0.4 * atr), 2)
            buy_high = round(min(cmp, ema20 * 1.015), 2) if cmp > ema20 * 1.015 else cmp

            min_target = round(cmp * 1.0525, 2)  # Guaranteed minimum >5% profit target
            tech_max_target = round(cmp + max(cmp * 0.09, 3.0 * atr), 2)
            if analyst_target and analyst_target > min_target:
                max_target = round(min(analyst_target, cmp * 1.22), 2)
                max_target = max(max_target, tech_max_target)
            else:
                max_target = tech_max_target

            max_growth_pct = round(((max_target - cmp) / cmp) * 100, 2)
            stop_loss = round(min(buy_low - 0.8 * atr, cmp * 0.965), 2)

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
                'Min Target': min_target,
                'Max Target': max_target,
                'Max Growth (%)': max_growth_pct,
                'Stop Loss': stop_loss,
                'RSI': rsi,
                'RVOL': vol_mult,
                'ADX': adx,
                '1Y Beta': round(beta, 2) if beta is not None else "",
                '52W High': high_52w,
                '52W Low': low_52w,
                '52W High Dist (%)': dist_52w,
                'Est Next EPS Gr (%)': est_eps_gr if est_eps_gr is not None else "",
                'Qtr Sales Gr (%)': round(qtr_rev_gr, 2) if qtr_rev_gr is not None else "",
                'Qtr Profit Gr (%)': round(qtr_eps_gr, 2) if qtr_eps_gr is not None else "",
                'Op Margin (%)': round(op_margin, 2) if op_margin is not None else "",
                'ROE (%)': round(roe, 2) if roe is not None else "",
                'PE Ratio': round(pe, 2) if pe is not None else "",
                'P/B Ratio': round(pb, 2) if pb is not None else "",
                'Debt/Equity': round(debt_eq, 2) if debt_eq is not None else "",
                'MCap (Cr)': mcap_cr
            }

            # 1. 30-Day Earnings Library: Strictly 1 to 35 Days Out
            if isinstance(days_left, int) and 1 <= days_left <= LIBRARY_MAX_DAYS:
                library_list.append(record)

            # 2. Strict Filter for "Quant-Fundy Logs" (Low-Phase Accumulation & Pre-Event Gems Only)
            if isinstance(days_left, int) and days_left <= 1:
                continue  # Never enter on Day 0 or Day 1
            if not (MIN_RSI <= rsi <= MAX_RSI):
                continue
            if chg_1w > MAX_1W_CHASE_PCT:
                continue  # Block stocks that have already run up >8.5% this week!

            # Must be near 20-EMA or 50-EMA support base (within 4.5%) so we buy LOW
            near_ema20 = -3.0 <= ((cmp - ema20) / ema20) * 100 <= 4.5 if ema20 > 0 else False
            near_ema50 = -2.0 <= ((cmp - ema50) / ema50) * 100 <= 4.5 if ema50 > 0 else False
            if not (near_ema20 or near_ema50):
                continue

            is_fin = any(w in sector.lower() for w in ['finance', 'bank', 'insurance', 'investment'])
            if qtr_rev_gr is not None and qtr_rev_gr < 0:
                continue
            if qtr_eps_gr is not None and qtr_eps_gr < 0:
                continue
            if not is_fin and op_margin is not None and op_margin <= 5.0:
                continue

            if ai_score >= 74:
                quant_setups.append(record)

        except Exception:
            continue

    quant_df = pd.DataFrame(quant_setups)
    if not quant_df.empty:
        quant_df = quant_df.sort_values(by=['AI Score', 'Est Next EPS Gr (%)'], ascending=[False, False])

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
        top_df = quant_df[['Ticker', 'Days Left', 'CMP', 'Buy Low', 'Min Target', 'Max Growth (%)']].head(12)
        msg = (
            f"<b>💎 V6.0 LOW-PHASE PRE-EVENT GEMS ({len(quant_df)} Setups)</b>\n"
            f"<i>Filtered for quiet base accumulation before results/events (Min +5.25% Target)</i>\n\n"
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
    print(f"Low-Phase Pre-Event Gems Found: {len(quant_df)} | 30-Day Catalysts: {len(library_list)}")
    if not quant_df.empty:
        print(quant_df[['Regime', 'AI Score', 'Ticker', 'Days Left', 'CMP', 'Buy Low', 'Min Target', 'Max Growth (%)']].head(12).to_string(index=False))
    push_notifications(quant_df, library_list, price_map)
