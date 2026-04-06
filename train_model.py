import pandas as pd
import ta
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import config

os.makedirs("models", exist_ok=True)

def add_indicators(df):
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
    return df

print("Loading data...")
all_data = []

for symbol in config.SYMBOLS:
    path = f"data/{symbol}.csv"
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue
    df = pd.read_csv(path)
    df["close"]  = df["close"].astype(float)
    df["volume"] = df["volume"].astype(float)
    if "high" not in df.columns:
        df["high"] = df["close"]
        df["low"]  = df["close"]
    df["high"] = df["high"].astype(float)
    df["low"]  = df["low"].astype(float)
    df = add_indicators(df)
    df["target"] = (df["close"].shift(-1) > df["close"]).astype(int)
    df.dropna(inplace=True)
    all_data.append(df)
    print(f"Loaded {symbol}: {len(df)} rows")

data = pd.concat(all_data)
print(f"\nTotal rows: {len(data)}")

FEATURES = [
    "rsi","macd","macd_sig","ema_9","ema_20","ema_50",
    "bb_high","bb_low","bb_mid","atr","adx","cci",
    "stoch","williams","momentum","volume"
]

X = data[FEATURES]
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("\nTraining improved AI model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test)) * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")
print("Old accuracy was: 54.55%")

if accuracy > 54.55:
    print(f"Improved by: +{accuracy-54.55:.2f}%")

joblib.dump(model, "models/trading_model.pkl")
print("Model saved!")
input("Press Enter to exit...")