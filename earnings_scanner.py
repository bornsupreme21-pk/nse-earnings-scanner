import yfinance as yf
import pandas as pd
import requests
import io
import datetime
import os
import json
import time
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings('ignore')

# --- Regime Parameters ---
LIBRARY_DAYS = 30            
QUANT_DAYS = 15              
MIN_RSI = 47.0               
MAX_RALLY_PCT = 15.0         
MIN_VOLUME_MULTIPLIER = 1.3  
EMA_PROXIMITY_PCT = 0.05     
MIN_PRICE_FILTER = 50.0      

# --- NSE Headers ---
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "*/*"
}

def get_all_nse_tickers():
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv"
    ]
    tickers = set()
    for url in urls:
        try:
            response = requests.get(url, headers=NSE_HEADERS, timeout=10)
            df = pd.read_csv(io.StringIO(response.text))
            df.columns = df.columns.str.strip()
            for symbol in df['Symbol']:
                tickers.add(f"{symbol}.NS")
        except: pass
    return list(tickers)

def get_nse_board_meetings_fallback():
    try:
        session = requests.Session()
        session.get("https://www.nseindia.com", headers=NSE_HEADERS, timeout=10)
        api_url = "https://www.nseindia.com/api/corporate-board-meetings"
        res = session.get(api_url, headers=NSE_HEADERS, timeout=10)
        data = res.json()
        fallback_dict = {}
        for item in data:
            if 'Financial Results' in item.get('purpose', ''):
                symbol = f"{item.get('symbol')}.NS"
                try: fallback_dict[symbol] = datetime.datetime.strptime(item.get('bm_date'), "%d-%b-%Y").date()
                except: pass
        return fallback_dict
    except: return {} 

def calculate_ai_score(rsi, vol_mult, dist_52w):
    score = 50 
    if rsi > 60: score += 10
    elif rsi > 50: score += 5
    if vol_mult > 3.0: score += 20
    elif vol_mult > 2.0: score += 15
    elif vol_mult > 1.5: score += 5
    if -5.0 <= dist_52w <= 0: score += 15
    elif -15.0 <= dist_52w < -5.0: score += 10
    
    score = min(max(score, 0), 100) 
    if score >= 75: decision = "🔥 STRONG BUY"
    elif score >= 60: decision = "✅ BUY"
    else: decision = "👀 WATCH"
    return score, decision

def execute_slow_scan(tickers):
    today = datetime.date.today()
    master_calendar, quant_setups = [], []
    
    print("Attempting to fetch NSE Board Meetings...")
    nse_dates = get_nse_board_meetings_fallback()

    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)['Close']
        if isinstance(nifty, pd.DataFrame): nifty = nifty.squeeze()
        nifty_1m_ret = (nifty.iloc[-1] - nifty.iloc[-21]) / nifty.iloc[-21]
    except: nifty_1m_ret = 0.0

    print(f"Executing Scan on {len(tickers)} stocks...")

    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i+batch_size]
        print(f"Processing Batch {i//batch_size + 1} of {(len(tickers)//batch_size)+1}...")
        
        try:
            data = yf.download(batch, period="1y", progress=False, group_by="ticker", threads=True)
        except Exception as e:
            print(f"Batch failed: {e}")
            time.sleep(2)
            continue
            
        if data.empty:
            time.sleep(2)
            continue

        for ticker in batch:
            try:
                # --- UNIVERSAL YFINANCE INDEX FIX ---
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(0): 
                        df = data[ticker].dropna(subset=['Close'])
                    elif ticker in data.columns.get_level_values(1):
                        df = data.xs(ticker, level=1, axis=1).dropna(subset=['Close'])
                    else:
                        continue
                else:
                    if len(batch) == 1: 
                        df = data.dropna(subset=['Close'])
                    else: 
                        continue
                        
                if len(df) < 50: 
                    continue
                
                c = df['Close'].squeeze() if isinstance(df['Close'], pd.DataFrame) else df['Close']
                v = df['Volume'].squeeze() if isinstance(df['Volume'], pd.DataFrame) else df['Volume']
                h = df['High'].squeeze() if isinstance(df['High'], pd.DataFrame) else df['High']
                l = df['Low'].squeeze() if isinstance(df['Low'], pd.DataFrame) else df['Low']
                
                cmp = round(c.iloc[-1], 2)
                if cmp < MIN_PRICE_FILTER: continue

                e_date = nse_dates.get(ticker)
                if e_date:
                    days_until_earnings = (e_date - today).days
                    if days_until_earnings < 0 or days_until_earnings > LIBRARY_DAYS: continue
                    earnings_date_str = str(e_date)
                else:
                    earnings_date_str, days_until_earnings = "Unknown", "N/A"

                cal_data = {
                    'Scan Date': str(today),
                    'Ticker': ticker.replace('.NS', ''),
                    'Sector': 'N/A', 
                    'Earnings Date': earnings_date_str,
                    'Days': days_until_earnings,
                    'CMP': cmp,
                    'Market Cap': 'N/A', 'Rev Growth': 'N/A', 'Earnings Growth': 'N/A', 'Div Yield': 'N/A'
                }
                master_calendar.append(cal_data)

                if days_until_earnings == "N/A" or days_until_earnings <= QUANT_DAYS:
                    ema20 = c.ewm(span=20, adjust=False).mean()
                    ema50 = c.ewm(span=50, adjust=False).mean()
                    vol_sma20 = v.rolling(20).mean()
                    
                    delta = c.diff()
                    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                    loss = -1 * delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
                    rsi_series = 100 - (100 / (1 + (gain / loss)))
                    
                    prev_c = c.shift(1)
                    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
                    atr_series = tr.rolling(14).mean()
                    
                    latest_rsi, latest_atr = rsi_series.iloc[-1], atr_series.iloc[-1]
                    latest_ema20, latest_ema50 = ema20.iloc[-1], ema50.iloc[-1]
                    
                    prev_month = c.iloc[-21] if len(c) > 21 else c.iloc[0]
                    monthly_return = ((cmp - prev_month) / prev_month) * 100
                    
                    recent_vol = v.iloc[-5:]
                    avg_vol = vol_sma20.iloc[-5:].mean()
                    vol_mult = recent_vol.max() / avg_vol if avg_vol > 0 else 0
                    dist_52w = ((cmp - h.max()) / h.max()) * 100
                    
                    if (MIN_RSI <= latest_rsi <= 78) and (monthly_return <= MAX_RALLY_PCT) and (vol_mult >= MIN_VOLUME_MULTIPLIER):
                        if (abs(cmp - latest_ema20) / latest_ema20 <= EMA_PROXIMITY_PCT or abs(cmp - latest_ema50) / latest_ema50 <= EMA_PROXIMITY_PCT):
                            if ((cmp - c.iloc[-21]) / c.iloc[-21]) >= nifty_returns:
                                ai_score, decision = calculate_ai_score(latest_rsi, vol_mult, dist_52w)
                                quant_data = {
                                    'Scan Date': str(today), 'Decision': decision, 'AI Score': ai_score,
                                    'Ticker': ticker.replace('.NS', ''), 'Sector': 'N/A',
                                    'Earnings Date': earnings_date_str, 'Days': days_until_earnings, 'CMP': cmp,
                                    'Target': round(cmp + max((cmp * 0.06), (2.5 * latest_atr)), 2),
                                    'Stop Loss': round(cmp - (1.5 * latest_atr), 2),
                                    'RSI': round(latest_rsi, 2), 'Vol Spike': f"{vol_mult:.1f}x",
                                    'PE Ratio': 'N/A', 'P/B Ratio': 'N/A', 'Debt/Equity': 'N/A',
                                    '52W High Dist (%)': round(dist_52w, 2)
                                }
                                quant_setups.append(quant_data)
            except Exception as e: 
                continue
                
        time.sleep(2)
            
    cal_df = pd.DataFrame(master_calendar)
    if not cal_df.empty:
        cal_df['SortDays'] = pd.to_numeric(cal_df['Days'], errors='coerce').fillna(999)
        cal_df = cal_df.sort_values(by='SortDays').drop(columns=['SortDays']).drop_duplicates(subset=['Ticker'])

    quant_df = pd.DataFrame(quant_setups)
    if not quant_df.empty:
        quant_df = quant_df.sort_values(by='AI Score', ascending=False).drop_duplicates(subset=['Ticker'])
        
    return cal_df, quant_df

def send_telegram_alert(dataframe):
    token, chat_id = os.getenv('TELEGRAM_BOT_TOKEN'), os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id or dataframe.empty: return
    tg_df = dataframe[['Decision', 'Ticker', 'CMP', 'Target', 'Stop Loss']]
    message = "🧠 *QUANT PREDICTOR ALERT*\n\n```text\n" + tg_df.to_string(index=False) + "\n```"
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"})

def push_to_webhook(data_type, dataframe):
    url = os.getenv('WEBHOOK_URL')
    if not url or dataframe.empty: return
    try:
        chunk_size = 200 
        records = dataframe.to_dict(orient='records')
        for idx, i in enumerate(range(0, len(records), chunk_size)):
            chunk = records[i:i + chunk_size]
            payload = {"type": data_type, "chunk_index": idx, "data": chunk}
            requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
            time.sleep(1.5) 
    except Exception as e:
        print(f"Webhook error ({data_type}): {e}")

if __name__ == "__main__":
    nse_watchlist = get_all_nse_tickers()
    if nse_watchlist:
        cal_df, quant_df = execute_slow_scan(nse_watchlist)
        if not cal_df.empty:
            print(f"Found {len(cal_df)} stocks. Pushing to Library...")
            push_to_webhook("calendar", cal_df)
        if not quant_df.empty:
            print(f"Found {len(quant_df)} setups. Pushing to Predictor & Telegram...")
            push_to_webhook("quant", quant_df)
            send_telegram_alert(quant_df)
        else:
            print("No actionable Quant-Fundy setups found today.")
