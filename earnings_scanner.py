import yfinance as yf
import pandas as pd
import requests
import io
import datetime
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Pro Quantitative Regime Parameters ---
DAYS_TO_EARNINGS = 15        # Extended to catch early accumulation
MIN_RSI = 47.0               # Lowered to catch momentum ignition
MAX_RALLY_PCT = 15.0         
MIN_VOLUME_MULTIPLIER = 1.3  # Adjusted for earlier footprint detection
EMA_PROXIMITY_PCT = 0.04
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

def calculate_ai_score(rsi, vol_mult, dist_52w, pe):
    """Machine Decision Predictor: Grades the setup from 0 to 100."""
    score = 50 # Base score
    
    # Momentum Factor
    if rsi > 60: score += 10
    elif rsi > 50: score += 5
    
    # Institutional Footprint (Smart Money Accumulation)
    if vol_mult > 3.0: score += 20
    elif vol_mult > 2.0: score += 15
    elif vol_mult > 1.5: score += 5
    
    # Breakout Proximity
    if -5.0 <= dist_52w <= 0: score += 15
    elif -15.0 <= dist_52w < -5.0: score += 10
    
    # Fundamental Buffer (Value protection)
    if 0 < pe < 30: score += 5
    elif pe > 80: score -= 10
    
    score = min(max(score, 0), 100) # Clamp between 0-100
    
    if score >= 80: decision = "🔥 STRONG BUY"
    elif score >= 65: decision = "✅ BUY"
    else: decision = "👀 WATCH"
    
    return score, decision

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
        
        if cmp < MIN_PRICE_FILTER:
            return None
        
        df['EMA20'] = close.ewm(span=20, adjust=False).mean()
        df['EMA50'] = close.ewm(span=50, adjust=False).mean()
        df['Vol_SMA20'] = volume.rolling(window=20).mean()
        
        # Wilder's RSI (14)
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = -1 * delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # ATR (14)
        high, low = df['High'], df['Low']
        close_prev = close.shift(1)
        tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        latest_rsi = df['RSI'].iloc[-1]
        latest_atr = df['ATR'].iloc[-1]
        latest_ema20 = df['EMA20'].iloc[-1]
        
        # --- FILTERS ---
        if not (MIN_RSI <= latest_rsi <= 78):
            return None

        prev_month = df.iloc[-21] if len(df) > 21 else df.iloc[0]
        if ((cmp - prev_month['Close']) / prev_month['Close']) * 100 > MAX_RALLY_PCT:
            return None
            
        recent_vol = volume.iloc[-5:]
        avg_vol = df['Vol_SMA20'].iloc[-5:].mean()
        vol_multiple = recent_vol.max() / avg_vol if avg_vol > 0 else 0
        if vol_multiple < MIN_VOLUME_MULTIPLIER:
            return None
            
        if not (abs(cmp - latest_ema20) / latest_ema20 <= EMA_PROXIMITY_PCT or abs(cmp - latest['EMA50']) / latest['EMA50'] <= EMA_PROXIMITY_PCT):
            return None
            
        if ((cmp - close.iloc[-21]) / close.iloc[-21]) < nifty_returns:
            return None
            
        # --- DATA GATHERING: Fundamentals & Prediction ---
        try:
            info = stock.info
            sector = info.get('sector', 'Unknown')
            pe = info.get('trailingPE', 0)
            pb = info.get('priceToBook', 0)
            debt_eq = info.get('debtToEquity', 0)
        except:
            sector, pe, pb, debt_eq = 'Unknown', 0, 0, 0
            
        high_52w = df['High'].max()
        dist_to_52w = ((cmp - high_52w) / high_52w) * 100
        
        # Machine Predictor Scoring
        ai_score, decision = calculate_ai_score(latest_rsi, vol_multiple, dist_to_52w, pe)
        
        stop_loss = round(cmp - (1.5 * latest_atr), 2)
        target = round(cmp + max((cmp * 0.06), (2.5 * latest_atr)), 2)
            
        return {
            'Scan Date': str(datetime.date.today()),
            'Decision': decision,
            'AI Score': ai_score,
            'Ticker': ticker.replace('.NS', ''),
            'Sector': sector,
            'Earnings Date': str(earnings_date),
            'Days': days_until_earnings,
            'CMP': cmp,
            'Target': target,
            'Stop Loss': stop_loss,
            'RSI': round(latest_rsi, 2),
            'Vol Spike': f"{vol_multiple:.1f}x",
            'PE Ratio': round(pe, 2) if pe else "N/A",
            'P/B Ratio': round(pb, 2) if pb else "N/A",
            'Debt/Equity': round(debt_eq, 2) if debt_eq else "N/A",
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
            nifty_1m_ret = 0.0
        else:
            nifty_close = nifty['Close']
            if isinstance(nifty_close, pd.DataFrame): nifty_close = nifty_close.squeeze()
            nifty_1m_ret = (nifty_close.iloc[-1] - nifty_close.iloc[-21]) / nifty_close.iloc[-21]
    except:
        nifty_1m_ret = 0.0
    
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(analyze_single_stock, t, today, nifty_1m_ret): t for t in tickers}
        for future in as_completed(futures):
            result = future.result()
            if result:
                valid_setups.append(result)
                
    if valid_setups:
        # Sort by best AI Score
        return pd.DataFrame(valid_setups).sort_values(by='AI Score', ascending=False)
    return pd.DataFrame()

def send_telegram_alert(dataframe):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not token or not chat_id:
        return

    # Keep Telegram ultra-clean for fast decision making
    tg_df = dataframe[['Decision', 'Ticker', 'CMP', 'Target', 'Stop Loss']]
    
    message = "🧠 *QUANT-FUNDY PREDICTOR ALERT*\n\n"
    message += f"```text\n{tg_df.to_string(index=False)}\n```"
    
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"}
    requests.post(url, json=payload)

def log_to_google_sheet_webhook(dataframe):
    webhook_url = os.getenv('WEBHOOK_URL')
    if not webhook_url:
        return

    try:
        payload = dataframe.to_dict(orient='records')
        headers = {'Content-Type': 'application/json'}
        requests.post(webhook_url, data=json.dumps(payload), headers=headers)
    except Exception as e:
        print(f"Webhook Error: {e}")

if __name__ == "__main__":
    nse_watchlist = get_all_nse_tickers()
    if nse_watchlist:
        found_setups = execute_market_scan(nse_watchlist)
        if not found_setups.empty:
            print(found_setups.to_string(index=False))
            send_telegram_alert(found_setups)
            log_to_google_sheet_webhook(found_setups)
        else:
            print("No stocks matched the strict Quant-Fundy criteria today.")
