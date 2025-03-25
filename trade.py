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

# -----------------------------
# Streamlit Config
# -----------------------------
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")
st.title("🚀 BTC/USDT AI Signal Dashboard")
st_autorefresh(interval=60000, key="refresh")  # refresh ทุก 60 วิ

# -----------------------------
# Columns for charts
# -----------------------------
chart_col1, chart_col2 = st.columns([2, 1])

# ========== STEP 1: Fetch Live Data ==============
try:
    exchange = ccxt.kucoin()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=100)
    df = pd.DataFrame(
        ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
except Exception as e:
    st.error(f"❌ Error fetching data: {e}")
    st.stop()

# ========== STEP 2: Feature Engineering ============
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

# ========== STEP 3: Create Target ================
df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

# ========== STEP 4: Prepare Data =================
df = df.dropna()
features = ['Return', 'Candle_Body', 'Range', 'EMA_fast', 'EMA_slow',
            'RSI', 'BollingerBands', 'OBV', 'ADX', 'Volume_Change']
X = df[features]
y = df['Target']

# ========== STEP 5: Train/Test Split =============
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# ========== STEP 6: Train Model ===================
model = XGBClassifier(n_estimators=100, max_depth=5,
                      learning_rate=0.1, eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== STEP 7: Predict Full & Confidence =====
proba = model.predict_proba(X)
df['Confidence'] = proba[:, 1]  # confidence of class 1 (price will go up)
confidence_threshold = 0.85
df['Signal'] = np.where(df['Confidence'] > confidence_threshold, 1,
                        np.where(df['Confidence'] < (1 - confidence_threshold), 0, np.nan))
df['Position'] = df['Signal'].map({1: 'Long', 0: 'Short'})
longs = df[df['Signal'] == 1]
shorts = df[df['Signal'] == 0]
neutral = df[df['Signal'].isna()]

# ========== STEP 8: Plot - Feature Importance =======
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

# ========== STEP 9: Plot - Confusion Matrix =========
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

# ========== STEP 10: Plot - BTC Price + Signals ======
price_fig = go.Figure()
price_fig.add_trace(go.Scatter(
    x=df.index, y=df['Close'], mode='lines', name='Price', line=dict(color='gray')))
price_fig.add_trace(go.Scatter(x=longs.index, y=longs['Close'], mode='markers',
                               name='Long', marker=dict(color='green', symbol='triangle-up', size=10)))
price_fig.add_trace(go.Scatter(x=shorts.index, y=shorts['Close'], mode='markers',
                               name='Short', marker=dict(color='red', symbol='triangle-down', size=10)))
price_fig.add_trace(go.Scatter(x=neutral.index, y=neutral['Close'], mode='markers',
                               name='No Signal', marker=dict(color='lightgray', size=6)))
price_fig.update_layout(
    title="📈 BTC Price with Long/Short Signals",
    xaxis_title="Time",
    yaxis_title="Price (USDT)",
    height=600,
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(orientation="h", yanchor="bottom",
                y=1.02, xanchor="right", x=1)
)

# ========== STEP 11: Show Charts in Streamlit =========
chart_col1.plotly_chart(price_fig, use_container_width=True)
chart_col2.plotly_chart(feature_fig, use_container_width=True)
chart_col2.plotly_chart(cm_fig, use_container_width=True)
