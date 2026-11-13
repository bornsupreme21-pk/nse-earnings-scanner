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
