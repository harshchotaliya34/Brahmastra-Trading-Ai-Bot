import streamlit as st
import ccxt
import pandas as pd
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- 1. ડેશબોર્ડ સેટઅપ ---
st.set_page_config(page_title="AI Trading Agent - Backtesting Edition", layout="wide")
st.title("📊 AI Trading Agent with Backtesting")

# --- 2. યુઝર ઇનપુટ ---
col1, col2, col3 = st.columns(3)
with col1:
    symbol = st.selectbox("ક્રિપ્ટો જોડી પસંદ કરો", ["BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT"])
with col2:
    timeframe = st.selectbox("ટાઈમફ્રેમ પસંદ કરો", ["5m", "15m", "1h", "4h", "1d"])
with col3:
    # બેકટેસ્ટિંગ માટે કેટલા ડેટાનો ઉપયોગ કરવો છે?
    limit = st.slider("બેકટેસ્ટિંગ ડેટા સાઈઝ (Candles)", min_value=500, max_value=2000, value=1000, step=100)

# --- 3. ડેટા ઇન્જેશન ---
@st.cache_data(ttl=60)
def fetch_data(symbol, timeframe, limit):
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(bars, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Time'] = pd.to_datetime(df['Time'], unit='ms')
    return df

with st.spinner('ડેટા ફેચ થઈ રહ્યો છે...'):
    df = fetch_data(symbol, timeframe, limit)

# --- 4. ફિચર એન્જિનિયરિંગ ---
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.sma(length=20, append=True)
df.ta.sma(length=50, append=True)
df.ta.atr(length=14, append=True)
df.dropna(inplace=True)

# --- 5. AI માટે Target બનાવવો ---
df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)

# --- 6. ડેટાને ટ્રેનિંગ અને ટેસ્ટિંગમાં વહેંચવો ---
split_index = int(len(df) * 0.8)
train_data = df.iloc[:split_index].dropna()
test_data = df.iloc[split_index:].copy()

features = ['RSI_14', 'MACD_12_26_9', 'SMA_20', 'SMA_50', 'ATRr_14']
X_train = train_data[features]
y_train = train_data['Target']
X_test = test_data[features]
y_test = test_data['Target']

# --- 7. મોડલ ટ્રેનિંગ ---
model = RandomForestClassifier(n_estimators=100, random_state=42)
with st.spinner('AI મોડલ ટ્રેન થઈ રહ્યું છે...'):
    model.fit(X_train, y_train)

# --- 8. બેકટેસ્ટિંગ લોજીક (Backtesting Logic) ---
st.header("📈 બેકટેસ્ટિંગ પરિણામો")

# ટેસ્ટ ડેટા પર મોડલની આગાહી
test_predictions = model.predict(X_test)
test_data['Prediction'] = test_predictions

initial_capital = 1000.0  
capital = initial_capital
position = 0  
buy_price = 0
trades = []

for index, row in test_data.iterrows():
    # Buy Signal
    if row['Prediction'] == 1 and position == 0:
        position = 1
        buy_price = row['Close']
        trade_entry_time = row['Time']
        
    # Sell Signal
    elif row['Prediction'] == 0 and position == 1:
        position = 0
        sell_price = row['Close']
        profit_loss = (sell_price - buy_price) / buy_price * 100 
        capital = capital * (1 + (profit_loss / 100))
        
        trades.append({
            'Entry Time': trade_entry_time,
            'Exit Time': row['Time'],
            'Buy Price': buy_price,
            'Sell Price': sell_price,
            'Profit/Loss (%)': round(profit_loss, 2),
            'Result': 'Win' if profit_loss > 0 else 'Loss'
        })

# --- 9. પરિણામોનું વિશ્લેષણ ---
total_trades = len(trades)
winning_trades = len([t for t in trades if t['Result'] == 'Win'])
losing_trades = len([t for t in trades if t['Result'] == 'Loss'])
win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
total_profit = capital - initial_capital

# મેટ્રિક્સ ડિસ્પ્લે
col1, col2, col3, col4 = st.columns(4)
col1.metric("શરૂઆતની રકમ", f"${initial_capital:.2f}")
col2.metric("અંતિમ રકમ (નફા/નુકસાન સાથે)", f"${capital:.2f}", f"${total_profit:.2f}")
col3.metric("કુલ ટ્રેડ્સ", total_trades)
col4.metric("Win Rate", f"{win_rate:.1f}%")

if total_trades > 0:
    st.subheader("વિગતવાર ટ્રેડ હિસ્ટ્રી")
    trades_df = pd.DataFrame(trades)
    
    def color_result(val):
        color = 'green' if val == 'Win' else 'red'
        return f'color: {color}'
        
    st.dataframe(trades_df.style.map(color_result, subset=['Result']))
else:
    st.warning("આ સમયગાળા દરમિયાન કોઈ ટ્રેડ લેવાયો નથી.")

# --- 10. વર્તમાન સિગ્નલ ---
st.divider()
st.subheader(f"🤖 લાઇવ માર્કેટ સિગ્નલ (છેલ્લી અપડેટ)")
latest_data = test_data[features].iloc[-1:]
latest_prediction = model.predict(latest_data)[0]
latest_close = df['Close'].iloc[-1]

if latest_prediction == 1:
    st.success(f"**BUY SIGNAL (ખરીદો)** - કિંમત: ${latest_close}")
else:
    st.error(f"**SELL SIGNAL (વેચો / રાહ જુઓ)** - કિંમત: ${latest_close}")2
