import yfinance as yf
import pandas as pd
import requests
import io
import datetime
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Pro Quantitative Regime Parameters ---
DAYS_TO_EARNINGS = 7
MAX_RALLY_PCT = 15.0         
MIN_VOLUME_MULTIPLIER = 1.5  
EMA_PROXIMITY_PCT = 0.03
MIN_PRICE_FILTER = 50.0      

def get_all_nse_tickers():
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
    try:
        stock = yf.Ticker(ticker)
        calendar = stock.calendar
        if calendar is None or calendar.empty:
            return None
            
        earnings_date = calendar.loc['Earnings Date'][0].date()
        days_until_earnings = (earnings_date - today).days
        
        if not (1 <= days_until_earnings <= DAYS_TO_EARNINGS):
            return None
            
        df = stock.history(period="1y")
        if len(df) < 50:
            return None
            
        close = df['Close']
        volume = df['Volume']
        latest = df.iloc[-1]
        cmp = round(latest['Close'], 2)
        
        # --- FILTER 1: Penny Stock Check ---
        if cmp < MIN_PRICE_FILTER:
            return None
        
        # --- TECHNICAL INDICATORS: EMAs, RSI, ATR ---
        df['EMA20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA50'] = close.ewm(span=50, adjust=False).mean()
        df['Vol_SMA20'] = volume.rolling(window=20).mean()
        
        # Wilder's RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = -1 * delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (Average True Range - 14) for Stop Loss & Target Math
        high, low = df['High'], df['Low']
        close_prev = close.shift(1)
        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        latest_rsi = df['RSI'].iloc[-1]
        latest_atr = df['ATR'].iloc[-1]
        latest_ema20 = df['EMA20'].iloc[-1]
        
        # --- FILTER 2: RSI Momentum Check ---
        # RSI should be in bullish momentum zone (above 50) but not heavily overbought (above 75)
        if not (50 <= latest_rsi <= 75):
            return None

        # --- FILTER 3: Overextension Check ---
        prev_month = df.iloc[-21] if len(df) > 21 else df.iloc[0]
        monthly_return = ((cmp - prev_month['Close']) / prev_month['Close']) * 100
        if monthly_return > MAX_RALLY_PCT:
            return None
            
        # --- FILTER 4: Volume Spike Check ---
        recent_vol = volume.iloc[-5:]
        avg_vol = df['Vol_SMA20'].iloc[-5:].mean()
        vol_multiple = recent_vol.max() / avg_vol if avg_vol > 0 else 0
        if vol_multiple < MIN_VOLUME_MULTIPLIER:
            return None
            
        # --- FILTER 5: EMA Support ---
        near_ema20 = abs(cmp - latest_ema20) / latest_ema20 <= EMA_PROXIMITY_PCT
        near_ema50 = abs(cmp - latest['EMA50']) / latest['EMA50'] <= EMA_PROXIMITY_PCT
        if not (near_ema20 or near_ema50):
            return None
            
        # --- FILTER 6: Relative Strength vs Nifty ---
        if ((cmp - close.iloc[-21]) / close.iloc[-21]) < nifty_returns:
            return None
            
        # --- QUANTITATIVE MATH: Targets, Zones, Stop-Loss ---
        try:
            sector = stock.info.get('sector', 'Unknown')
        except:
            sector = 'Unknown'
            
        high_52w = df['High'].max()
        dist_to_52w = ((cmp - high_52w) / high_52w) * 100
        
        # Stop-Loss: 1.5x Average True Range below CMP (Accounts for daily volatility)
        stop_loss = round(cmp - (1.5 * latest_atr), 2)
        
        # Target: Minimum 5% profit OR 2x ATR breakout profit (whichever is larger)
        target = round(cmp + max((cmp * 0.05), (2.0 * latest_atr)), 2)
        
        # Buy Zone: Between 20 EMA and CMP
        buy_zone_bottom = round(min(cmp, latest_ema20), 2)
        buy_zone = f"₹{buy_zone_bottom} - ₹{cmp}"
        
        # Hold Duration: Hold until Earnings Day or 1 day after
        hold_duration = f"{days_until_earnings} to {days_until_earnings + 1} Days"
            
        return {
            'Ticker': ticker.replace('.NS', ''),
            'Sector': sector,
            'Earnings Date': str(earnings_date),
            'Hold Duration': hold_duration,
            'CMP': cmp,
            'Buy Zone': buy_zone,
            'Target': target,
            'Stop Loss': stop_loss,
            'RSI': round(latest_rsi, 2),
            'Vol Spike': f"{vol_multiple:.1f}x",
            '52W High Dist (%)': round(dist_to_52w, 2)
        }
    except Exception:
        return None

def execute_market_scan(tickers):
    today = datetime.date.today()
    valid_setups = []
    
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)
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
        # Sort by best RSI momentum
        return pd.DataFrame(valid_setups).sort_values(by='RSI', ascending=False)
    return pd.DataFrame()

def send_telegram_alert(dataframe):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        print("Telegram credentials missing. Print to console only.")
        return

    # Send concise version to Telegram to avoid screen clutter
    tg_df = dataframe[['Ticker', 'Buy Zone', 'Target', 'Stop Loss', 'Hold Duration']]
    
    message = "🎯 *PRO PRE-EARNINGS SETUPS*\n\n"
    message += f"```text\n{tg_df.to_string(index=False)}\n```"
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"}
    requests.post(url, json=payload)

def log_to_google_sheet_webhook(dataframe):
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
        else:
            print(f"Failed to log to Google Sheets. Status: {response.status_code}")
    except Exception as e:
        print(f"Error connecting to Webhook: {e}")

if __name__ == "__main__":
    nse_watchlist = get_all_nse_tickers()
    if nse_watchlist:
        found_setups = execute_market_scan(nse_watchlist)
        if not found_setups.empty:
            print(found_setups.to_string(index=False))
            send_telegram_alert(found_setups)
            log_to_google_sheet_webhook(found_setups)
        else:
            print("No stocks matched the strict pre-earnings criteria today.")
