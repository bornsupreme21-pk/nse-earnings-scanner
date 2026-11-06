import yfinance as yf
import pandas as pd
import requests
import io
import datetime
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Regime Parameters ---
LIBRARY_DAYS = 30            
QUANT_DAYS = 15              
MIN_RSI = 47.0               
MAX_RALLY_PCT = 15.0         
MIN_VOLUME_MULTIPLIER = 1.3  
EMA_PROXIMITY_PCT = 0.05     
MIN_PRICE_FILTER = 50.0      

# --- NSE Fallback Headers ---
NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5"
}

def get_all_nse_tickers():
    """Fetches Nifty 500 + Nifty Microcap 250 (750 Total Stocks) to balance scope and rate limits."""
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
        except Exception as e:
            print(f"Failed to fetch list from {url}: {e}")
    return list(tickers)

def get_nse_board_meetings_fallback():
    """Fetches the live board meetings JSON directly from NSE as a fallback database."""
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
                date_str = item.get('bm_date') 
                try:
                    fallback_dict[symbol] = datetime.datetime.strptime(date_str, "%d-%b-%Y").date()
                except:
                    pass
        return fallback_dict
    except Exception:
        return {} 

def format_percentage(val):
    if val is None or pd.isna(val): return "N/A"
    return f"{round(val * 100, 2)}%"

def calculate_ai_score(rsi, vol_mult, dist_52w, pe):
    score = 50 
    if rsi > 60: score += 10
    elif rsi > 50: score += 5
    if vol_mult > 3.0: score += 20
    elif vol_mult > 2.0: score += 15
    elif vol_mult > 1.5: score += 5
    if -5.0 <= dist_52w <= 0: score += 15
    elif -15.0 <= dist_52w < -5.0: score += 10
    if 0 < pe < 30: score += 5
    elif pe > 80: score -= 10
    score = min(max(score, 0), 100) 
    
    if score >= 80: decision = "🔥 STRONG BUY"
    elif score >= 65: decision = "✅ BUY"
    else: decision = "👀 WATCH"
    return score, decision

def analyze_single_stock(ticker, today, nifty_returns, nse_fallback_data):
    try:
        stock = yf.Ticker(ticker)
        
        # --- DUAL-ENGINE EARNINGS FETCH ---
        earnings_date_str, days_until_earnings = "Unknown", "N/A"
        
        # Engine 1: Yahoo Finance
        try:
            cal = stock.calendar
            if isinstance(cal, dict) and 'Earnings Date' in cal:
                e_date = cal['Earnings Date'][0].date()
                if (e_date - today).days >= 0:
                    earnings_date_str, days_until_earnings = str(e_date), (e_date - today).days
            elif hasattr(cal, 'empty') and not cal.empty and 'Earnings Date' in cal.index:
                e_date = cal.loc['Earnings Date'][0].date()
                if (e_date - today).days >= 0:
                    earnings_date_str, days_until_earnings = str(e_date), (e_date - today).days
        except: pass
        
        # Engine 2: NSE Official API Fallback
        if days_until_earnings == "N/A" and ticker in nse_fallback_data:
            e_date = nse_fallback_data[ticker]
            if (e_date - today).days >= 0:
                earnings_date_str, days_until_earnings = str(e_date), (e_date - today).days

        # Library Filter
        if days_until_earnings != "N/A" and days_until_earnings > LIBRARY_DAYS:
            return None
            
        df = stock.history(period="1y")
        if len(df) < 50: return None
            
        close, volume = df['Close'], df['Volume']
        latest_cmp = round(close.iloc[-1], 2)
        if latest_cmp < MIN_PRICE_FILTER: return None

        # Fetch Fundamentals Once
        try:
            info = stock.info
            sector = info.get('sector', 'Unknown')
            mcap = round(info.get('marketCap', 0) / 10000000, 2) if info.get('marketCap') else "N/A" 
            rev_growth = format_percentage(info.get('revenueGrowth'))
            earn_growth = format_percentage(info.get('earningsGrowth'))
            div_yield = format_percentage(info.get('dividendYield'))
            pe = info.get('trailingPE', 0)
            pb = info.get('priceToBook', 0)
            debt_eq = info.get('debtToEquity', 0)
        except:
            sector, mcap, rev_growth, earn_growth, div_yield, pe, pb, debt_eq = 'Unknown', "N/A", "N/A", "N/A", "N/A", 0, 0, 0

        cal_data = {
            'Scan Date': str(datetime.date.today()),
            'Ticker': ticker.replace('.NS', ''),
            'Sector': sector,
            'Earnings Date': earnings_date_str,
            'Days': days_until_earnings,
            'CMP': latest_cmp,
            'Market Cap': mcap,
            'Rev Growth': rev_growth,
            'Earnings Growth': earn_growth,
            'Div Yield': div_yield
        }
        
        # --- QUANTITATIVE PREDICTOR MATH ---
        quant_data = None
        if days_until_earnings == "N/A" or days_until_earnings <= QUANT_DAYS:
            df['EMA20'] = close.ewm(span=20, adjust=False).mean()
            df['EMA50'] = close.ewm(span=50, adjust=False).mean()
            df['Vol_SMA20'] = volume.rolling(window=20).mean()
            
            delta = close.diff()
            rs = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean() / -1 * delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
            df['RSI'] = 100 - (100 / (1 + rs))
            
            high, low, close_prev = df['High'], df['Low'], close.shift(1)
            tr = pd.concat([high - low, (high - close_prev).abs(), (low - close_prev).abs()], axis=1).max(axis=1)
            df['ATR'] = tr.rolling(14).mean()
            
            latest_rsi, latest_atr, latest_ema20 = df['RSI'].iloc[-1], df['ATR'].iloc[-1], df['EMA20'].iloc[-1]
            prev_month = df.iloc[-21] if len(df) > 21 else df.iloc[0]
            monthly_return = ((latest_cmp - prev_month['Close']) / prev_month['Close']) * 100
            
            recent_vol = volume.iloc[-5:]
            avg_vol = df['Vol_SMA20'].iloc[-5:].mean()
            vol_mult = recent_vol.max() / avg_vol if avg_vol > 0 else 0
            
            dist_52w = ((latest_cmp - df['High'].max()) / df['High'].max()) * 100
            
            if (MIN_RSI <= latest_rsi <= 78) and (monthly_return <= MAX_RALLY_PCT) and (vol_mult >= MIN_VOLUME_MULTIPLIER):
                if (abs(latest_cmp - latest_ema20) / latest_ema20 <= EMA_PROXIMITY_PCT or abs(latest_cmp - df['EMA50'].iloc[-1]) / df['EMA50'].iloc[-1] <= EMA_PROXIMITY_PCT):
                    if ((latest_cmp - close.iloc[-21]) / close.iloc[-21]) >= nifty_returns:
                        
                        ai_score, decision = calculate_ai_score(latest_rsi, vol_mult, dist_52w, pe)
                        
                        quant_data = {
                            'Scan Date': str(datetime.date.today()),
                            'Decision': decision,
                            'AI Score': ai_score,
                            'Ticker': ticker.replace('.NS', ''),
                            'Sector': sector,
                            'Earnings Date': earnings_date_str,
                            'Days': days_until_earnings,
                            'CMP': latest_cmp,
                            'Target': round(latest_cmp + max((latest_cmp * 0.06), (2.5 * latest_atr)), 2),
                            'Stop Loss': round(latest_cmp - (1.5 * latest_atr), 2),
                            'RSI': round(latest_rsi, 2),
                            'Vol Spike': f"{vol_mult:.1f}x",
                            'PE Ratio': round(pe, 2) if pe else "N/A",
                            'P/B Ratio': round(pb, 2) if pb else "N/A",
                            'Debt/Equity': round(debt_eq, 2) if debt_eq else "N/A",
                            '52W High Dist (%)': round(dist_52w, 2)
                        }

        return {'calendar': cal_data, 'quant': quant_data}
    except Exception:
        return None

def execute_market_scan(tickers):
    today = datetime.date.today()
    master_calendar = []
    quant_setups = []
    
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)['Close']
        if isinstance(nifty, pd.DataFrame): nifty = nifty.squeeze()
        nifty_1m_ret = (nifty.iloc[-1] - nifty.iloc[-21]) / nifty.iloc[-21]
    except: nifty_1m_ret = 0.0

    print("Fetching NSE Board Meetings Fallback Data...")
    nse_fallback_data = get_nse_board_meetings_fallback()
    print(f"Loaded {len(nse_fallback_data)} fallback dates from NSE.")
    
    # Process in carefully sized batches to completely avoid Yahoo Finance 429 Errors
    batch_size = 50
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {executor.submit(analyze_single_stock, t, today, nifty_1m_ret, nse_fallback_data): t for t in batch}
            for future in as_completed(futures):
                res = future.result()
                if res:
                    master_calendar.append(res['calendar'])
                    if res['quant']: quant_setups.append(res['quant'])
        # A vital 2-second pause between batches keeps the server invisible to Yahoo's security
        time.sleep(2.0)
                
    cal_df = pd.DataFrame(master_calendar)
    if not cal_df.empty:
        cal_df['SortDays'] = pd.to_numeric(cal_df['Days'], errors='coerce').fillna(999)
        cal_df = cal_df.sort_values(by='SortDays').drop(columns=['SortDays'])

    quant_df = pd.DataFrame(quant_setups)
    if not quant_df.empty:
        quant_df = quant_df.sort_values(by='AI Score', ascending=False)
        
    return cal_df, quant_df

def send_telegram_alert(dataframe):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id or dataframe.empty: return

    tg_df = dataframe[['Decision', 'Ticker', 'CMP', 'Target', 'Stop Loss']]
    message = "🧠 *QUANT-FUNDY PREDICTOR ALERT*\n\n"
    message += f"```text\n{tg_df.to_string(index=False)}\n```"
    requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": message, "parse_mode": "MarkdownV2"})

def push_to_webhook(data_type, dataframe):
    url = os.getenv('WEBHOOK_URL')
    if not url or dataframe.empty: return
    try:
        chunk_size = 500
        records = dataframe.to_dict(orient='records')
        for i in range(0, len(records), chunk_size):
            chunk = records[i:i + chunk_size]
            payload = {"type": data_type, "data": chunk}
            requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
    except Exception as e:
        print(f"Webhook error ({data_type}): {e}")

if __name__ == "__main__":
    nse_watchlist = get_all_nse_tickers()
    if nse_watchlist:
        cal_df, quant_df = execute_market_scan(nse_watchlist)
        
        if not cal_df.empty:
            print(f"Found {len(cal_df)} stocks for the 30-Day Earnings Library.")
            push_to_webhook("calendar", cal_df)
            
        if not quant_df.empty:
            print(f"Found {len(quant_df)} actionable Quant-Fundy setups.")
            push_to_webhook("quant", quant_df)
            send_telegram_alert(quant_df)
        else:
            print("No stocks passed the strict Quant-Fundy filter today.")
