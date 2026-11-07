import yfinance as yf
import pandas as pd
import requests
import io
import datetime
import os
import json
import warnings
import time

warnings.filterwarnings('ignore')

# --- Regime Parameters ---
QUANT_DAYS = 15              
MIN_RSI = 47.0               
MAX_RALLY_PCT = 15.0         
MIN_VOLUME_MULTIPLIER = 1.3  
EMA_PROXIMITY_PCT = 0.05     
MIN_PRICE_FILTER = 50.0      

def get_all_nse_tickers():
    """Fetches Nifty 500 + Microcap 250 lists."""
    print("Fetching Ticker Universe...")
    urls = [
        "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
        "https://archives.nseindia.com/content/indices/ind_niftymicrocap250_list.csv"
    ]
    tickers = set()
    for url in urls:
        try:
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            df = pd.read_csv(io.StringIO(res.text))
            df.columns = df.columns.str.strip()
            for symbol in df['Symbol']:
                tickers.add(f"{symbol}.NS")
        except:
            pass
    # Fallback just in case NSE archives are blocked
    if not tickers:
        tickers = {"RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"}
    return list(tickers)

def calculate_ai_score(rsi, vol_mult, dist_52w, pe):
    score = 50 
    if rsi > 60: score += 10
    elif rsi > 50: score += 5
    if vol_mult > 3.0: score += 20
    elif vol_mult > 2.0: score += 15
    elif vol_mult > 1.5: score += 5
    if -5.0 <= dist_52w <= 0: score += 15
    elif -15.0 <= dist_52w < -5.0: score += 10
    
    # Fundamental Buffer
    if pe != "N/A":
        if 0 < pe < 30: score += 5
        elif pe > 80: score -= 10
        
    score = min(max(score, 0), 100) 
    if score >= 75: decision = "🔥 STRONG BUY"
    elif score >= 60: decision = "✅ BUY"
    else: decision = "👀 WATCH"
    return score, decision

def execute_smart_scan(tickers):
    today = datetime.date.today()
    
    try:
        nifty = yf.download("^NSEI", period="1y", progress=False)['Close']
        nifty_1m_ret = (nifty.iloc[-1] - nifty.iloc[-21]) / nifty.iloc[-21]
    except: 
        nifty_1m_ret = 0.0

    print(f"Phase 1: Downloading raw price data for {len(tickers)} stocks...")
    
    # Split into two batches to guarantee no download timeouts
    midpoint = len(tickers) // 2
    batch1, batch2 = tickers[:midpoint], tickers[midpoint:]
    
    df1 = yf.download(batch1, period="1y", progress=False, group_by="ticker", threads=True)
    df2 = yf.download(batch2, period="1y", progress=False, group_by="ticker", threads=True)
    
    # Combine the downloaded data
    data = pd.concat([df1, df2], axis=1)

    print("Phase 2: Running local quantitative math (0 API calls)...")
    technically_primed_tickers = []
    quant_metrics = {}

    for ticker in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(0): continue
                c = data[ticker]['Close'].dropna()
                v = data[ticker]['Volume'].dropna()
                h = data[ticker]['High'].dropna()
                l = data[ticker]['Low'].dropna()
            else:
                continue # Safety skip if data structure fails
                
            if len(c) < 50: continue
            
            cmp = round(c.iloc[-1], 2)
            if cmp < MIN_PRICE_FILTER: continue

            ema20 = c.ewm(span=20, adjust=False).mean()
            ema50 = c.ewm(span=50, adjust=False).mean()
            vol_sma20 = v.rolling(20).mean()
            
            delta = c.diff()
            gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
            loss = -1 * delta.clip(upper=0).ewm(alpha=1/14, adjust=False).mean()
            rsi = 100 - (100 / (1 + (gain / loss)))
            
            prev_c = c.shift(1)
            tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            latest_rsi, latest_atr = rsi.iloc[-1], atr.iloc[-1]
            latest_ema20, latest_ema50 = ema20.iloc[-1], ema50.iloc[-1]
            
            prev_month = c.iloc[-21] if len(c) > 21 else c.iloc[0]
            monthly_return = ((cmp - prev_month) / prev_month) * 100
            
            recent_vol = v.iloc[-5:]
            avg_vol = vol_sma20.iloc[-5:].mean()
            vol_mult = recent_vol.max() / avg_vol if avg_vol > 0 else 0
            
            dist_52w = ((cmp - h.max()) / h.max()) * 100
            
            # THE GATES: Only pass if momentum is perfect
            if (MIN_RSI <= latest_rsi <= 78) and (monthly_return <= MAX_RALLY_PCT) and (vol_mult >= MIN_VOLUME_MULTIPLIER):
                if (abs(cmp - latest_ema20) / latest_ema20 <= EMA_PROXIMITY_PCT or abs(cmp - latest_ema50) / latest_ema50 <= EMA_PROXIMITY_PCT):
                    if ((cmp - c.iloc[-21]) / c.iloc[-21]) >= nifty_1m_ret:
                        
                        # Save metrics for Phase 3
                        technically_primed_tickers.append(ticker)
                        quant_metrics[ticker] = {
                            'CMP': cmp, 'RSI': latest_rsi, 'ATR': latest_atr, 
                            'Vol Mult': vol_mult, 'Dist 52W': dist_52w
                        }
        except:
            continue

    print(f"Phase 3: Found {len(technically_primed_tickers)} momentum breakouts.")
    print("Fetching Earnings Dates & Fundamentals ONLY for these winners...")
    
    final_setups = []
    
    # We only ping Yahoo for the tiny list of winners. Unblockable.
    for ticker in technically_primed_tickers:
        time.sleep(0.5) # Gentle pause
        try:
            stock = yf.Ticker(ticker)
            
            # Get Dates
            e_date_str, days_to_earn = "Unknown", "N/A"
            try:
                cal = stock.calendar
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    e_date = cal['Earnings Date'][0].date()
                    if (e_date - today).days >= 0:
                        e_date_str, days_to_earn = str(e_date), (e_date - today).days
                elif hasattr(cal, 'empty') and not cal.empty and 'Earnings Date' in cal.index:
                    e_date = cal.loc['Earnings Date'][0].date()
                    if (e_date - today).days >= 0:
                        e_date_str, days_to_earn = str(e_date), (e_date - today).days
            except: pass

            # Ensure it fits the 15-day window (or is unknown)
            if days_to_earn != "N/A" and days_to_earn > QUANT_DAYS:
                continue

            # Get Fundamentals
            try:
                info = stock.info
                sector = info.get('sector', 'Unknown')
                pe = round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else "N/A"
                pb = round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else "N/A"
                debt_eq = round(info.get('debtToEquity', 0), 2) if info.get('debtToEquity') else "N/A"
            except:
                sector, pe, pb, debt_eq = 'Unknown', "N/A", "N/A", "N/A"

            # Retrieve saved math
            metrics = quant_metrics[ticker]
            cmp, atr = metrics['CMP'], metrics['ATR']
            
            ai_score, decision = calculate_ai_score(metrics['RSI'], metrics['Vol Mult'], metrics['Dist 52W'], pe)

            setup = {
                'Scan Date': str(today),
                'Decision': decision,
                'AI Score': ai_score,
                'Ticker': ticker.replace('.NS', ''),
                'Sector': sector,
                'Earnings Date': e_date_str,
                'Days': days_to_earn,
                'CMP': cmp,
                'Target': round(cmp + max((cmp * 0.06), (2.5 * atr)), 2),
                'Stop Loss': round(cmp - (1.5 * atr), 2),
                'RSI': round(metrics['RSI'], 2),
                'Vol Spike': f"{metrics['Vol Mult']:.1f}x",
                'PE Ratio': pe,
                'P/B Ratio': pb,
                'Debt/Equity': debt_eq,
                '52W High Dist (%)': round(metrics['Dist 52W'], 2)
            }
            final_setups.append(setup)
        except Exception as e:
            print(f"Failed to fetch info for {ticker}: {e}")
            continue

    df = pd.DataFrame(final_setups)
    if not df.empty:
        df = df.sort_values(by='AI Score', ascending=False)
    return df

def push_results(dataframe):
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    webhook = os.getenv('WEBHOOK_URL')

    if dataframe.empty:
        print("No setups passed the final phase today.")
        return

    # 1. Telegram
    if token and chat_id:
        tg_df = dataframe[['Decision', 'Ticker', 'CMP', 'Target', 'Stop Loss']]
        msg = "🧠 *QUANT PREDICTOR ALERT*\n\n```text\n" + tg_df.to_string(index=False) + "\n```"
        requests.post(f"https://api.telegram.org/bot{token}/sendMessage", json={"chat_id": chat_id, "text": msg, "parse_mode": "MarkdownV2"})
        print("Telegram Sent.")

    # 2. Google Sheets
    if webhook:
        payload = {"type": "quant", "chunk_index": 1, "data": dataframe.to_dict(orient='records')}
        res = requests.post(webhook, json=payload, headers={'Content-Type': 'application/json'})
        if res.status_code == 200:
            print("Google Sheet Updated.")
        else:
            print(f"Sheet Error: {res.status_code}")

if __name__ == "__main__":
    nse_watchlist = get_all_nse_tickers()
    if nse_watchlist:
        final_df = execute_smart_scan(nse_watchlist)
        push_results(final_df)
