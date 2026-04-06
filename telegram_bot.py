from binance.client import Client
from binance.exceptions import BinanceAPIException
import joblib, ta, time
import pandas as pd
from datetime import datetime
import config

try:
    from telegram_bot import send_signal, send_trade
    TELEGRAM = bool(config.TELEGRAM_TOKEN)
except:
    TELEGRAM = False

print("Connecting to Binance Testnet...")
client = Client(config.API_KEY, config.API_SECRET, testnet=True)
print("Connected!")
model = joblib.load("models/trading_model.pkl")
print("AI Model loaded!")

FEATURES = [
    "rsi","macd","macd_sig","ema_9","ema_20","ema_50",
    "bb_high","bb_low","bb_mid","atr","adx","cci",
    "stoch","williams","momentum","volume"
]

positions = {}

def get_balance(asset):
    try:
        b = client.get_asset_balance(asset=asset)
        return float(b["free"]) if b else 0
    except:
        return 0

def get_price(symbol):
    return float(client.get_symbol_ticker(symbol=symbol)["price"])

def place_buy(symbol):
    try:
        price = get_price(symbol)
        qty   = round(config.TRADE_USDT / price, 5)
        client.create_order(
            symbol=symbol,
            side="BUY",
            type="MARKET",
            quantity=qty
        )
        print(f"✅ BUY {symbol} @ ${price}")
        if TELEGRAM:
            send_trade(symbol, "BUY", price, config.TRADE_USDT)
        return price
    except BinanceAPIException as e:
        print(f"BUY error: {e.message}")
        return None

def place_sell(symbol):
    try:
        price = get_price(symbol)
        qty   = round(config.TRADE_USDT / price, 5)
        client.create_order(
            symbol=symbol,
            side="SELL",
            type="MARKET",
            quantity=qty
        )
        print(f"✅ SELL {symbol} @ ${price}")
        if TELEGRAM:
            send_trade(symbol, "SELL", price, config.TRADE_USDT)
        return price
    except BinanceAPIException as e:
        print(f"SELL error: {e.message}")
        return None

def get_signal(symbol):
    try:
        klines = client.get_klines(symbol=symbol, interval="1h", limit=100)
        df = pd.DataFrame(klines, columns=[
            "time","open","high","low","close","volume",
            "ct","qv","tr","tb","tq","ig"
        ])
        for col in ["open","high","low","close","volume"]:
            df[col] = df[col].astype(float)
        df["rsi"]      = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["macd"]     = ta.trend.MACD(df["close"]).macd()
        df["macd_sig"] = ta.trend.MACD(df["close"]).macd_signal()
        df["ema_9"]    = ta.trend.EMAIndicator(df["close"], 9).ema_indicator()
        df["ema_20"]   = ta.trend.EMAIndicator(df["close"], 20).ema_indicator()
        df["ema_50"]   = ta.trend.EMAIndicator(df["close"], 50).ema_indicator()
        df["bb_high"]  = ta.volatility.BollingerBands(df["close"]).bollinger_hband()
        df["bb_low"]   = ta.volatility.BollingerBands(df["close"]).bollinger_lband()
        df["bb_mid"]   = ta.volatility.BollingerBands(df["close"]).bollinger_mavg()
        df["atr"]      = ta.volatility.AverageTrueRange(df["high"],df["low"],df["close"]).average_true_range()
        df["adx"]      = ta.trend.ADXIndicator(df["high"],df["low"],df["close"]).adx()
        df["cci"]      = ta.trend.CCIIndicator(df["high"],df["low"],df["close"]).cci()
        df["stoch"]    = ta.momentum.StochasticOscillator(df["high"],df["low"],df["close"]).stoch()
        df["williams"] = ta.momentum.WilliamsRIndicator(df["high"],df["low"],df["close"]).williams_r()
        df["momentum"] = df["close"].pct_change(5)
        df.dropna(inplace=True)
        latest     = df[FEATURES].iloc[-1:]
        proba      = model.predict_proba(latest)[0]
        pred       = model.predict(latest)[0]
        confidence = round(max(proba)*100, 1)
        signal     = "BUY" if pred==1 else "SELL"
        price      = round(df["close"].iloc[-1], 2)
        rsi        = round(df["rsi"].iloc[-1], 2)
        return signal, price, rsi, confidence
    except Exception as e:
        print(f"Signal error {symbol}: {e}")
        return None, None, None, None

def run():
    print("\n"+"="*50)
    print("   NEXUSAI AUTO TRADING BOT")
    print(f"   Coins:  {config.SYMBOLS}")
    print(f"   Amount: ${config.TRADE_USDT} USDT each")
    print(f"   SL: {config.STOP_LOSS}% | TP: {config.TAKE_PROFIT}%")
    print(f"   Min Confidence: {config.MIN_CONFIDENCE}%")
    print("="*50+"\n")

    confirm = input("Type YES to start auto trading: ")
    if confirm.strip().upper() != "YES":
        print("Cancelled. Stay safe!")
        return

    print("\nBot started! Checking every 60 seconds...\n")

    while True:
        print(f"\n{datetime.now().strftime('%H:%M:%S')} Checking signals...")
        usdt = get_balance("USDT")
        print(f"Balance: ${usdt:.2f} USDT")
        print(f"Active positions: {list(positions.keys())}")

        for symbol in config.SYMBOLS:
            signal, price, rsi, confidence = get_signal(symbol)
            if not signal:
                continue

            print(f"  {symbol}: {signal} ${price} RSI:{rsi} Conf:{confidence}%")

            # Check stop loss / take profit
            if symbol in positions:
                entry = positions[symbol]
                sl = entry * (1 - config.STOP_LOSS / 100)
                tp = entry * (1 + config.TAKE_PROFIT / 100)
                if price <= sl:
                    print(f"🛑 STOP LOSS hit {symbol}!")
                    place_sell(symbol)
                    del positions[symbol]
                    continue
                if price >= tp:
                    print(f"🎉 TAKE PROFIT hit {symbol}!")
                    place_sell(symbol)
                    del positions[symbol]
                    continue

            # BUY signal
            if signal == "BUY" and confidence >= config.MIN_CONFIDENCE:
                if symbol not in positions:
                    if usdt >= config.TRADE_USDT:
                        entry = place_buy(symbol)
                        if entry:
                            positions[symbol] = entry
                            usdt -= config.TRADE_USDT
                    else:
                        print(f"Not enough USDT for {symbol}")

            # SELL signal
            elif signal == "SELL" and confidence >= config.MIN_CONFIDENCE:
                if symbol in positions:
                    place_sell(symbol)
                    del positions[symbol]

        print(f"\nWaiting 60 seconds...")
        time.sleep(60)

if __name__ == "__main__":
    run()