import yfinance as yf
import pandas as pd
import requests
import io
import datetime
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Configuration & Regime Parameters ---
DAYS_TO_EARNINGS = 7
MAX_RALLY_PCT = 15.0         
MIN_VOLUME_MULTIPLIER = 1.5  
EMA_PROXIMITY_PCT = 0.03     

def get_all_nse_tickers():
    """Fetches the live list of all active NSE equities directly from NSE India."""
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        df = pd.read_csv(io.StringIO(response.text))
        df.columns = df.columns.str.strip()
        df_eq = df[df['SERIES'] == 'EQ']
        return [f"{symbol}.NS" for symbol in df_eq['SYMBOL']]
    except Exception as e:
        print(f"Failed to fetch NSE master list: {e}")
        return []

def analyze_single_stock(ticker, today, nifty_returns):
    """Processes a single stock to check if it meets the pre-earnings criteria."""
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is None or calendar.empty:
            return None
            
        earnings_date = calendar.loc['Earnings Date'][0].date()
        days_until_earnings = (earnings_date - today).days
        
        if not (1 <= days_until_earnings <= DAYS_TO_EARNINGS):
            return None
            
        df = stock.history(period="6mo")
        if len(df) < 50:
            return None
            
        close = df['Close']
        volume = df['Volume']
        
        df['EMA20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA50'] = close.ewm(span=50, adjust=False).mean()
        df['Vol_SMA20'] = volume.rolling(window=20).mean()
        
        latest = df.iloc[-1]
        prev_month = df.iloc[-21] if len(df) > 21 else df.iloc[0]
        
        monthly_return = ((latest['Close'] - prev_month['Close']) / prev_month['Close']) * 100
        if monthly_return > MAX_RALLY_PCT:
            return None
            
        recent_vol = volume.iloc[-5:]
        recent_vol_sma = df['Vol_SMA20'].iloc[-5:]
        if not any(recent_vol > (recent_vol_sma * MIN_VOLUME_MULTIPLIER)):
            return None
            
        near_ema20 = abs(latest['Close'] - latest['EMA20']) / latest['EMA20'] <= EMA_PROXIMITY_PCT
        near_ema50 = abs(latest['Close'] - latest['EMA50']) / latest['EMA50'] <= EMA_PROXIMITY_PCT
        if not (near_ema20 or near_ema50):
            return None
            
        if ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21]) < nifty_returns:
            return None
            
        return {
            'Ticker': ticker.replace('.NS', ''),
            'Earnings': str(earnings_date),
            'Days': days_until_earnings,
            'CMP': round(latest['Close'], 2),
            '1M Rally (%)': round(monthly_return, 2),
            'Support': '20 EMA' if near_ema20 else '50 EMA'
        }
    except Exception:
        return None

def execute_market_scan(tickers):
    today = datetime.date.today()
    valid_setups = []
    
    # Robust fallback if Yahoo Finance blocks the NIFTY 50 download
    try:
        nifty = yf.download("^NSEI", period="6mo", progress=False)
        if nifty.empty or 'Close' not in nifty or len(nifty) < 21:
            print("Warning: NIFTY 50 download failed. Defaulting benchmark to 0%.")
            nifty_1m_ret = 0.0
        else:
            nifty_close = nifty['Close']
            if isinstance(nifty_close, pd.DataFrame):
                nifty_close = nifty_close.squeeze()
            nifty_1m_ret = (nifty_close.iloc[-1] - nifty_close.iloc[-21]) / nifty_close.iloc[-21]
    except Exception as e:
        print(f"Warning: NIFTY 50 error ({e}). Defaulting benchmark to 0%.")
        nifty_1m_ret = 0.0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_single_stock, t, today, nifty_1m_ret): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                valid_setups.append(result)
                
    if valid_setups:
        return pd.DataFrame(valid_setups).sort_values(by='Days')
    return pd.DataFrame()

def send_telegram_alert(dataframe):
    """Sends the formatted dataframe results to your Telegram."""
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("Telegram credentials missing. Print to console only.")
        return

    message = "🎯 *PRIME PRE-EARNINGS SETUPS*\n\n"
    message += f"```text\n{dataframe.to_string(index=False)}\n```"
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"}
    requests.post(url, json=payload)

if __name__ == "__main__":
    nse_watchlist = get_all_nse_tickers()
    if nse_watchlist:
        found_setups = execute_market_scan(nse_watchlist)
        if not found_setups.empty:
            print(found_setups.to_string(index=False))
            send_telegram_alert(found_setups)
        else:
            print("No stocks matched the strict pre-earnings criteria today.")
            import json

def log_to_google_sheet_webhook(dataframe):
    """Sends scan results to a dedicated Google Sheet via Webhook."""
    webhook_url = os.getenv('WEBHOOK_URL')
    if not webhook_url:
        print("Webhook URL missing. Skipping Sheets export.")
        return

    try:
        dataframe.insert(0, 'Scan Date', str(datetime.date.today()))
        payload = dataframe.to_dict(orient='records')
        headers = {'Content-Type': 'application/json'}
        
        response = requests.post(webhook_url, data=json.dumps(payload), headers=headers)
        
        if response.status_code == 200:
            print("Successfully logged data to the Google Sheet.")
            if __name__ == "__main__":
    nse_watchlist = get_all_nse_tickers()
    if nse_watchlist:
        found_setups = execute_market_scan(nse_watchlist)
        if not found_setups.empty:
            print(found_setups.to_string(index=False))
            send_telegram_alert(found_setups)
            log_to_google_sheet_webhook(found_setups) # <--- ADDED HERE
        else:
            print("No stocks matched the strict pre-earnings criteria today.")
        else:
            print(f"Failed to log to Google Sheets. Status: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to Webhook: {e}")
