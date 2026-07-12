import streamlit as st
import ccxt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# --- 1. મેન્યુઅલ ગણતરી માટેના ફંક્શન ---
def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    ema_gain = gain.ewm(com=period - 1, adjust=False).mean()
    ema_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = ema_gain / ema_loss
    return 100 - (100 / (1 + rs))

def calculate_sma(df, period=20):
    return df['Close'].rolling(window=period).mean()

# --- 2. ડેશબોર્ડ સેટઅપ ---
st.set_page_config(page_title="AI Trading Agent", layout="wide")
st.title("📊 AI Trading Agent - Fast Mode")

# --- 3. યુઝર ઇનપુટ ---
symbol = st.selectbox("ક્રિપ્ટો જોડી:", ["BTC/USDT", "ETH/USDT", "SOL/USDT"])
timeframe = st.selectbox("ટાઈમફ્રેમ:", ["5m", "15m", "1h", "1d"])
limit = st.slider("ડેટા સાઈઝ:", min_value=500, max_value=2000, value=1000)

# --- 4. ડેટા ઇન્જેશન ---
@st.cache_data(ttl=60)
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    return df

df = fetch_data(symbol, timeframe, limit)

# --- 5. ફિચર એન્જિનિયરિંગ (મેન્યુઅલ ગણતરી) ---
df['RSI_14'] = calculate_rsi(df)
df['SMA_20'] = calculate_sma(df, 20)
df['SMA_50'] = calculate_sma(df, 50)
df.dropna(inplace=True)

# --- 6. AI માટે Target અને ટ્રેનિંગ ---
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
split_index = int(len(df) * 0.8)
train_data = df.iloc[:split_index].dropna()
test_data = df.iloc[split_index:].copy()

features = ['RSI_14', 'SMA_20', 'SMA_50']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(train_data[features], train_data['Target'])

# --- 7. બેકટેસ્ટિંગ અને રિઝલ્ટ ---
st.header("📈 બેકટેસ્ટિંગ પરિણામો")
test_data['Prediction'] = model.predict(test_data[features])

capital = 1000.0
position = 0
buy_price = 0
trades = []

for index, row in test_data.iterrows():
    if row['Prediction'] == 1 and position == 0:
        position = 1
        buy_price = row['Close']
    elif row['Prediction'] == 0 and position == 1:
        position = 0
        profit_loss = (row['Close'] - buy_price) / buy_price * 100
        capital = capital * (1 + (profit_loss / 100))
        trades.append({'Profit/Loss (%)': round(profit_loss, 2)})

st.metric("Win Rate", f"{(len([t for t in trades if t['Profit/Loss (%)'] > 0]) / len(trades) * 100 if len(trades) > 0 else 0):.1f}%")
st.metric("કુલ નફો/નુકસાન", f"${capital - 1000:.2f}")

# --- 8. લાઇવ સિગ્નલ ---
st.divider()
st.subheader("🤖 લાઇવ સિગ્નલ")
latest_pred = model.predict(df[features].iloc[-1:])[0]
st.success("BUY SIGNAL" if latest_pred == 1 else "SELL SIGNAL")
