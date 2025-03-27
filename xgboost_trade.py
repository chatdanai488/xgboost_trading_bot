import pandas as pd
import numpy as np
import ta
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import plotly.graph_objects as go
import plotly.express as px
import ccxt

st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")
st.title("🚀 BTC/USDT AI Signal Dashboard")
st_autorefresh(interval=5000, key="refresh")

chart_col1, chart_col2 = st.columns([2, 1])

try:
    exchange = ccxt.kucoin()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='5m', limit=100)
    df = pd.DataFrame(
        ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

df['Candle_Body'] = abs(df['Close'] - df['Open'])
df['Range'] = df['High'] - df['Low']
df['Return'] = df['Close'].pct_change()
df['EMA_fast'] = df['Close'].ewm(span=5).mean()
df['EMA_slow'] = df['Close'].ewm(span=20).mean()
df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
df['BollingerBands'] = ta.volatility.BollingerBands(
    df['Close']).bollinger_lband()
df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close']).adx()
df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
    df['Close'], df['Volume']).on_balance_volume()
df['Volume_Change'] = df['Volume'].pct_change()

df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
df = df.dropna()
features = ['Return', 'Candle_Body', 'Range', 'EMA_fast', 'EMA_slow',
            'RSI', 'BollingerBands', 'OBV', 'ADX', 'Volume_Change']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)
model = XGBClassifier(n_estimators=100, max_depth=5,
                      learning_rate=0.1, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

proba = model.predict_proba(X)
df['Confidence'] = proba[:, 1]
confidence_threshold = 0.80
df['Signal'] = np.where(df['Confidence'] > confidence_threshold, 1,
                        np.where(df['Confidence'] < (1 - confidence_threshold), 0, np.nan))
df['Position'] = df['Signal'].map({1: 'Long', 0: 'Short'})
longs = df[df['Signal'] == 1]
shorts = df[df['Signal'] == 0]
neutral = df[df['Signal'].isna()]

# Normalize Confidence for scaling with price
conf_min = df['Confidence'].min()
conf_max = df['Confidence'].max()
price_min = df['Close'].min()
price_max = df['Close'].max()
df['Confidence_scaled'] = ((df['Confidence'] - conf_min) / (conf_max - conf_min)) \
    * (price_max - price_min) + price_min

importances = model.feature_importances_
feat_names = X.columns.tolist()
sorted_idx = np.argsort(importances)
sorted_feats = [feat_names[i] for i in sorted_idx]
sorted_imports = importances[sorted_idx]

feature_fig = go.Figure()
feature_fig.add_trace(go.Bar(
    x=sorted_imports,
    y=sorted_feats,
    orientation='h',
    marker_color='steelblue'
))
feature_fig.update_layout(
    title="🔍 Feature Importance (XGBoost)",
    xaxis_title="Importance",
    yaxis_title="Feature",
    height=400,
    margin=dict(l=40, r=20, t=40, b=40)
)

cm = confusion_matrix(y_test, y_pred)
cm_fig = px.imshow(
    cm,
    text_auto=True,
    color_continuous_scale='RdBu',
    labels=dict(x="Predicted", y="Actual"),
    x=['No Rise (0)', 'Rise (1)'],
    y=['No Rise (0)', 'Rise (1)']
)
cm_fig.update_layout(
    title="🎯 Confusion Matrix",
    height=400,
    margin=dict(l=40, r=20, t=40, b=40)
)

price_fig = go.Figure()
price_fig.add_trace(go.Scatter(
    x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='gray')))
price_fig.add_trace(go.Scatter(
    x=df.index, y=df['EMA_fast'], mode='lines', name='EMA_fast', line=dict(color='orange')))
price_fig.add_trace(go.Scatter(
    x=df.index, y=df['EMA_slow'], mode='lines', name='EMA_slow', line=dict(color='blue')))
price_fig.add_trace(go.Scatter(x=df.index, y=df['Confidence_scaled'], mode='lines',
                               name='Confidence', line=dict(color='purple', dash='dot')))
price_fig.add_trace(go.Scatter(x=longs.index, y=longs['Close'], mode='markers',
                               name='Long', marker=dict(color='green', symbol='triangle-up', size=10)))
price_fig.add_trace(go.Scatter(x=shorts.index, y=shorts['Close'], mode='markers',
                               name='Short', marker=dict(color='red', symbol='triangle-down', size=10)))
price_fig.add_trace(go.Scatter(x=neutral.index, y=neutral['Close'], mode='markers',
                               name='No Signal', marker=dict(color='lightgray', size=6)))
price_fig.update_layout(
    title="📈 BTC Price with Long/Short Signals + EMA + Confidence",
    xaxis_title="Time",
    yaxis_title="Price (USDT)",
    height=600,
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom",
                y=1.02, xanchor="right", x=1)
)

# RSI subplot
rsi_fig = go.Figure()
rsi_fig.add_trace(go.Scatter(
    x=df.index, y=df['RSI'], name='RSI', line=dict(color='cyan')))
rsi_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
                  line=dict(color='red', dash='dash'))
rsi_fig.add_shape(type='line', x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
                  line=dict(color='green', dash='dash'))
rsi_fig.update_layout(title="RSI Indicator", yaxis_title="RSI", height=300)

# Backtest PnL (simple strategy)
df['Shifted_Close'] = df['Close'].shift(-5)
df['Trade_Return'] = np.where(df['Signal'].notna(),
                              (df['Shifted_Close'] - df['Close']) / df['Close'], 0)
df['Strategy_Return'] = df['Trade_Return'] * np.where(df['Signal'] == 1, 1, -1)
df['Cumulative_Return'] = (1 + df['Strategy_Return']).cumprod()

pnl_fig = go.Figure()
pnl_fig.add_trace(go.Scatter(
    x=df.index, y=df['Cumulative_Return'], name='Strategy PnL', line=dict(color='gold')))
pnl_fig.update_layout(title="💰 Backtest Cumulative PnL",
                      height=300, yaxis_title="Cumulative Return")

# Show in Streamlit
chart_col1.plotly_chart(price_fig, use_container_width=True)
chart_col1.plotly_chart(rsi_fig, use_container_width=True)
chart_col1.plotly_chart(pnl_fig, use_container_width=True)
chart_col2.plotly_chart(feature_fig, use_container_width=True)
chart_col2.plotly_chart(cm_fig, use_container_width=True)
