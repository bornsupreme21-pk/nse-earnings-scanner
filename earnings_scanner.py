import os
import json
import datetime
import requests
import pandas as pd
from tradingview_screener import Query, Column

# --- Quantitative Regime Parameters ---
MIN_PRICE_FILTER = 50.0
MIN_RSI = 47.0
MAX_RSI = 78.0
MIN_VOL_SPIKE = 1.3
EMA_PROXIMITY_PCT = 0.05

def calculate_ai_score(rsi, vol_mult, dist_52w, pe):
    score = 50
    if rsi >= 60: score += 15
    elif rsi >= 52: score += 10
    
    if vol_mult >= 2.5: score += 20
    elif vol_mult >= 1.8: score += 15
    elif vol_mult >= 1.3: score += 10
    
    if -5.0 <= dist_52w <= 0: score += 15
    elif -12.0 <= dist_52w < -5.0: score += 10
    
    if isinstance(pe, (int, float)) and 0 < pe < 35:
        score += 5
        
    score = min(max(score, 0), 100)
    if score >= 80: decision = "🔥 STRONG BUY"
    elif score >= 65: decision = "✅ BUY"
    else: decision = "👀 WATCH"
    return score, decision

def fetch_nse_market_data():
    """Pulls live technicals & fundamentals for top 750 NSE stocks in 1 request (Zero 429 blocks)."""
    print("Querying Institutional Screener Engine for top 750 NSE stocks...")
    q = (Query()
         .set_markets('india')
         .select(
             'name', 'sector', 'close', 'volume', 'average_volume_30d_calc',
             'RSI', 'EMA20', 'EMA50', 'ATR', 'price_52_week_high',
             'price_earnings_ttm', 'price_book_fq', 'debt_to_equity', 'market_cap_basic'
         )
         .where(
             Column('exchange') == 'NSE',
             Column('close') >= MIN_PRICE_FILTER,
             Column('average_volume_30d_calc') > 50000
         )
         .order_by('market_cap_basic', ascending=False)
         .limit(750))
    
    total_count, df = q.get_scanner_data()
    print(f"Successfully downloaded {len(df)} liquid NSE stocks.")
    return df

def process_scanner_data(df):
    today_str = str(datetime.date.today())
    quant_setups = []
    library_list = []

    for _, row in df.iterrows():
        try:
            ticker = str(row.get('name', '')).strip()
            sector = str(row.get('sector') or 'Unknown')
            cmp = round(float(row['close']), 2)
            
            vol = float(row.get('volume') or 0)
            avg_vol = float(row.get('average_volume_30d_calc') or 1)
            vol_mult = round(vol / avg_vol, 2) if avg_vol > 0 else 0.0
            
            rsi = round(float(row.get('RSI') or 0), 2)
            ema20 = float(row.get('EMA20') or cmp)
            ema50 = float(row.get('EMA50') or cmp)
            atr = float(row.get('ATR') or (cmp * 0.02))
            high_52w = float(row.get('price_52_week_high') or cmp)
            
            dist_52w = round(((cmp - high_52w) / high_52w) * 100, 2) if high_52w > 0 else 0.0
            
            pe = round(float(row['price_earnings_ttm']), 2) if pd.notna(row.get('price_earnings_ttm')) else "N/A"
            pb = round(float(row['price_book_fq']), 2) if pd.notna(row.get('price_book_fq')) else "N/A"
            debt_eq = round(float(row['debt_to_equity']), 2) if pd.notna(row.get('debt_to_equity')) else "N/A"

            # Add top 250 stocks to Market Library tab regardless of filters
            if len(library_list) < 250:
                library_list.append({
                    'Scan Date': today_str, 'Ticker': ticker, 'Sector': sector,
                    'CMP': cmp, 'RSI': rsi, 'Vol Spike': f"{vol_mult:.1f}x",
                    'PE Ratio': pe, 'P/B Ratio': pb, 'Debt/Equity': debt_eq,
                    '52W High Dist (%)': dist_52w
                })

            # --- Strict Quant-Fundy Momentum Filters ---
            if not (MIN_RSI <= rsi <= MAX_RSI):
                continue
            if vol_mult < MIN_VOL_SPIKE:
                continue
                
            near_ema20 = abs(cmp - ema20) / ema20 <= EMA_PROXIMITY_PCT if ema20 > 0 else False
            near_ema50 = abs(cmp - ema50) / ema50 <= EMA_PROXIMITY_PCT if ema50 > 0 else False
            if not (near_ema20 or near_ema50):
                continue

            ai_score, decision = calculate_ai_score(rsi, vol_mult, dist_52w, pe)
            
            buy_low = round(min(cmp, ema20), 2)
            buy_zone = f"₹{buy_low} - ₹{cmp}"
            target = round(cmp + max(cmp * 0.06, 2.5 * atr), 2)
            stop_loss = round(cmp - (1.5 * atr), 2)

            quant_setups.append({
                'Scan Date': today_str,
                'Decision': decision,
                'AI Score': ai_score,
                'Ticker': ticker,
                'Sector': sector,
                'CMP': cmp,
                'Buy Zone': buy_zone,
                'Target': target,
                'Stop Loss': stop_loss,
                'RSI': rsi,
                'Vol Spike': f"{vol_mult:.1f}x",
                'PE Ratio': pe,
                'P/B Ratio': pb,
                'Debt/Equity': debt_eq,
                '52W High Dist (%)': dist_52w
            })
        except Exception:
            continue

    quant_df = pd.DataFrame(quant_setups)
    if not quant_df.empty:
        quant_df = quant_df.sort_values(by=['AI Score', 'RSI'], ascending=[False, False])
    else:
        # Heartbeat row so the sheet NEVER stays blank
        quant_df = pd.DataFrame([{
            'Scan Date': today_str, 'Decision': '⏸️ NO MATCH TODAY', 'AI Score': 0,
            'Ticker': 'SYSTEM_ACTIVE', 'Sector': 'Market Scan Complete', 'CMP': 0,
            'Buy Zone': 'N/A', 'Target': 0, 'Stop Loss': 0, 'RSI': 0,
            'Vol Spike': '0.0x', 'PE Ratio': 'N/A', 'P/B Ratio': 'N/A',
            'Debt/Equity': 'N/A', '52W High Dist (%)': 0
        }])

    return quant_df, library_list

def push_notifications(quant_df, library_list):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    webhook = os.getenv('WEBHOOK_URL')

    # 1. Push to Google Sheets with Verbose Verification
    if webhook:
        payload = {
            "clear_main": False,
            "quant": quant_df.to_dict(orient='records'),
            "library": library_list
        }
        try:
            res = requests.post(webhook, json=payload, headers={'Content-Type': 'application/json'}, timeout=30)
            print(f"Google Sheets Webhook Status: {res.status_code} | Response: {res.text[:200]}")
        except Exception as e:
            print(f"Google Sheets Webhook Failed: {e}")
    else:
        print("ERROR: WEBHOOK_URL secret is empty!")

    # 2. Push to Telegram (Using safe HTML mode)
    if token and chat_id:
        top_df = quant_df[['Decision', 'Ticker', 'CMP', 'Target', 'Stop Loss']].head(15)
        msg = f"<b>🧠 QUANT-FUNDY SCANNER REPORT ({len(quant_df)} Setups)</b>\n\n<pre>{top_df.to_string(index=False)}</pre>"
        try:
            tg_res = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                timeout=15
            )
            print(f"Telegram Status: {tg_res.status_code} | Response: {tg_res.text[:200]}")
        except Exception as e:
            print(f"Telegram Failed: {e}")
    else:
        print("ERROR: TELEGRAM secrets are missing!")

if __name__ == "__main__":
    raw_df = fetch_nse_market_data()
    quant_df, library_list = process_scanner_data(raw_df)
    print(quant_df.head(10).to_string(index=False))
    push_notifications(quant_df, library_list)
