import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import ta  # technical indicators
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import ccxt
import sys
import matplotlib.ticker as ticker
import streamlit as st
from streamlit_autorefresh import st_autorefresh
# sys.stdout.reconfigure(encoding='utf-8')

# -----------------------------
# Streamlit Config
# -----------------------------
st.title("🚀 BTC/USDT AI Signal Dashboard")

st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")
st_autorefresh(interval=60000, key="refresh")  # refresh ทุก 5 วินาที


# -----------------------------
# Placeholder for dynamic charts
# -----------------------------
chart_col1, chart_col2 = st.columns([2, 1])
price_chart_placeholder = chart_col1.empty()
feature_chart_placeholder = chart_col2.empty()

# -----------------------------
# เปิดโหมด interactive plot
# -----------------------------
# plt.ion()

# เตรียม figure & axes
# Feature importance + Confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2, ax_price = plt.subplots(figsize=(14, 6))   # Price chart


# ========== STEP 1: ดึงข้อมูล ==============
exchange = ccxt.kucoin()
ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=100)
df = pd.DataFrame(
    ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
df.set_index('Timestamp', inplace=True)

# ========== STEP 2: สร้าง Features ===========
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

# ========== STEP 3: สร้าง Target ============
df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)
# target_threshold = 0.001  # ขึ้นเกิน 0.1%
# future_periods = 5  # 5 นาทีข้างหน้า
# df['Target'] = ((df['Close'].shift(-future_periods) -
# df['Close']) / df['Close'] > target_threshold).astype(int)

# ========== STEP 4: เตรียมข้อมูล =============
df = df.dropna()
features = ['Return', 'Candle_Body', 'Range', 'EMA_fast', 'EMA_slow',
            'RSI', 'BollingerBands', 'OBV', 'ADX', 'Volume_Change']
X = df[features]
y = df['Target']

# ========== STEP 5: Train/Test Split =========
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# ========== STEP 6: ฝึกโมเดล XGBoost =========
model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1,
                      eval_metric='logloss')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ========== STEP 7: วาดกราฟ =============
importances = model.feature_importances_
feat_names = X.columns.tolist()
cm = confusion_matrix(y_test, y_pred)

# Predict ทั้ง df เพื่อดูจุดเข้า
# df['Signal'] = model.predict(X)
# ค่านี้จะให้ความมั่นใจต่อ class 0 และ class 1
proba = model.predict_proba(X)  # shape = (n, 2)
df['Confidence'] = proba[:, 1]  # ความมั่นใจใน class 1 (จะขึ้น)
print(df['Confidence'])
confidence_threshold = 0.975

df['Signal'] = np.where(df['Confidence'] > confidence_threshold, 1,
                        np.where(df['Confidence'] < (1 - confidence_threshold), 0, np.nan))

df['Position'] = df['Signal'].map({1: 'Long', 0: 'Short'})

# กรองสัญญาณ
longs = df[df['Signal'] == 1]
shorts = df[df['Signal'] == 0]
neutral = df[df['Signal'].isna()]  # จุดไม่มีสัญญาณ

# ----- วาด Feature Importance แบบเรียงจากมากไปน้อย -----
axes[0].clear()
sorted_idx = np.argsort(importances)
sorted_feats = [feat_names[i] for i in sorted_idx]
sorted_imports = importances[sorted_idx]

axes[0].barh(sorted_feats, sorted_imports, color='steelblue')
axes[0].set_title("🔍 Feature Importance (XGBoost)", fontsize=12)
axes[0].set_xlabel("Importance", fontsize=10)
axes[0].xaxis.set_major_formatter(
    ticker.FuncFormatter(lambda x, _: f'{x:.2f}'))
axes[0].grid(axis='x', linestyle='--', alpha=0.5)

# ----- วาด Confusion Matrix พร้อม label สวยงาม -----
axes[1].clear()
cm_labels = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm", cbar=False, square=True, ax=axes[1],
            annot_kws={"size": 14})

axes[1].set_xlabel("Predicted", fontsize=10)
axes[1].set_ylabel("Actual", fontsize=10)
axes[1].set_title("🎯 Confusion Matrix", fontsize=12)
axes[1].xaxis.tick_top()  # ย้าย label ไปด้านบน
axes[1].xaxis.set_label_position('top')
# เขียนคำอธิบายแต่ละตำแหน่งใน matrix
description_text = """
Meaning of position:
11 → Predicted to go up and went up (True Positive)
10 → Predicted not to go up but went up (False Negative)
01 → Predicted to go up but didn't go up (False Positive)
00 → Predicted not to go up and didn't go up (True Negative)
"""

# เพิ่มเป็นข้อความใต้กราฟ
axes[1].text(0.5, -0.4, description_text, fontsize=10,
             ha='center', va='top', transform=axes[1].transAxes)

# ----- วาด Chart ราคา + สัญญาณ -----
ax_price.clear()
ax_price.plot(df.index, df['Close'], color='gray', label='Price')
ax_price.scatter(
    longs.index, longs['Close'], marker='^', color='green', label='Long Signal')
ax_price.scatter(
    shorts.index, shorts['Close'], marker='v', color='red', label='Short Signal')
# จุด Neutral (ไม่มีสัญญาณ)
ax_price.scatter(
    neutral.index, neutral['Close'], marker='o', color='lightgray', s=10, label='No Signal')

ax_price.set_title("Long/Short Signals from Model")
ax_price.set_xlabel("Timestamp")
ax_price.set_ylabel("Price")
ax_price.legend()
ax_price.grid(True)

# ----- แสดงกราฟทั้งหมด -----
# plt.tight_layout()
# plt.draw()
# plt.pause(5)  # refresh ทุก 5 วินาที

# -----------------------------
# แสดงผลใน Streamlit
# -----------------------------
feature_chart_placeholder.pyplot(fig)
price_chart_placeholder.pyplot(fig2)
