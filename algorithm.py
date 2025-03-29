import pandas as pd
import numpy as np
import lightgbm as lgb
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator
from stable_baselines3 import PPO
import gym
from gym import spaces
import ccxt
from ta.trend import MACD
from ta.volatility import BollingerBands
import ta
import os
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import sys
import streamlit as st
import time
from linebot.v3.messaging import MessagingApi, Configuration, ApiClient
from linebot.v3.messaging.models import TextMessage, PushMessageRequest, ImageMessage
sys.stdout.reconfigure(encoding='utf-8')

user_id = 'U9713684916c46794109094dc4c180a08'  # 替換成您的 User ID
config = Configuration(
    access_token='GwwVLTOUtUrelOSZZgmA6Zv2JMbFOafVVSOfMw7WeKZloPTxFx1xa3CndCqnrjdIKBUZw8FnbGZuZAU3nxaI2ouAUJpRMxXv8wBRdnmrECfuRvMNH1oO/MolAWgFdGMaL6PLTT7SvZxm815y3RgznAdB04t89/1O/w1cDnyilFU=')
messaging_api = MessagingApi(ApiClient(configuration=config))

# Streamlit User Interface
st.set_page_config(
    page_title="📈 Crypto Trading Signal Display", layout="centered")
st.title("Crypto Trading Signal Display")

# Create a placeholder for dynamic updates
placeholder = st.empty()

while True:

    exchange = ccxt.kucoin()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1m', limit=100)

    df = pd.DataFrame(
        ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])

    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')

    df['Volume_Change'] = df['Volume'].pct_change()
    df['Candle_Body'] = abs(df['Close'] - df['Open'])
    df['Range'] = df['High'] - df['Low']
    df['Return'] = df['Close'].pct_change()

    df['RSI'] = RSIIndicator(close=df['Close'].squeeze(), window=14).rsi()
    df['EMA_fast'] = df['Close'].ewm(span=5).mean()
    df['EMA_slow'] = df['Close'].ewm(span=20).mean()
    df['BollingerBands'] = ta.volatility.BollingerBands(
        df['Close'].squeeze()).bollinger_lband()
    df['ADX'] = ta.trend.ADXIndicator(
        df['High'].squeeze(), df['Low'].squeeze(), df['Close'].squeeze()).adx()
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(
        df['Close'].squeeze(), df['Volume'].squeeze()).on_balance_volume()

    # ---------- STEP 3: สร้าง Label ----------

    def compute_trend_direction_and_duration(df, sideway_thresh=0.001, max_lookahead=20):
        result = []
        for i in range(len(df)):
            direction = None
            duration = 0
            for j in range(1, max_lookahead + 1):
                if i + j >= len(df):
                    break
                ret = (df['Close'].iloc[i + j] - df['Close'].iloc[i]) / \
                    df['Close'].iloc[i]
                if abs(ret) < sideway_thresh:
                    if direction is None:
                        direction = "sideway"
                    elif direction != "sideway":
                        break
                elif ret > 0:
                    if direction is None:
                        direction = "up"
                    elif direction != "up":
                        break
                elif ret < 0:
                    if direction is None:
                        direction = "down"
                    elif direction != "down":
                        break
                duration += 1
            result.append((direction, duration))
        df['TrendDir'], df['TrendLen'] = zip(*result)
        return df

    df = compute_trend_direction_and_duration(df)
    df.dropna(inplace=True)

    # ---------- STEP 4: เตรียมฟีเจอร์และ Target ----------

    df['TrendDirEncoded'] = LabelEncoder().fit_transform(
        df['TrendDir'])  # up=2, sideway=1, down=0

    features = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'RSI', 'EMA_fast', 'EMA_slow',
        'BollingerBands', 'ADX', 'OBV',
        'Volume_Change', 'Candle_Body', 'Range', 'Return'
    ]

    X = df[features]
    y_dir = df['TrendDirEncoded']
    y_len = df['TrendLen']

    # ---------- STEP 5: Train Models ----------

    X_train, X_test, y_dir_train, y_dir_test = train_test_split(
        X, y_dir, test_size=0.2, random_state=42)
    _, _, y_len_train, y_len_test = train_test_split(
        X, y_len, test_size=0.2, random_state=42)

    model_dir = LGBMClassifier()
    model_dir.fit(X_train, y_dir_train)

    model_len = LGBMRegressor()
    model_len.fit(X_train, y_len_train)

    # === STEP 6: ทำนายแท่งล่าสุด ===
    latest = X.iloc[[-1]]
    dir_pred = model_dir.predict(latest)[0]
    proba = model_dir.predict_proba(latest)[0][dir_pred]
    len_pred = int(model_len.predict(latest)[0])
    dir_label = ['down', 'sideway', 'up'][dir_pred]
    adx_val = df['ADX'].iloc[-1]

    # === STEP 7: เงื่อนไขกลยุทธ์เลือกเทรดแบบละเอียด ===

    # กำหนดสถานะเริ่มต้นถ้ายังไม่มี
    try:
        position
    except NameError:
        position = None
        entry_price = 0

    # ดึงราคาล่าสุดของแท่งเทียนปัจจุบัน (สมมติ df ถูกกำหนดแล้ว)
    price_now = df['Close'].iloc[-1]

    if dir_label in ['up', 'down'] and len_pred >= 5 and proba > 0.85 and adx_val > 25:
        # จุดเริ่มต้นการเข้า position ใหม่
        if position is None:
            if dir_label == 'up':
                action = f"📈 จุดเริ่มต้น: *เริ่ม Long* (~{len_pred} แท่ง)"
            else:
                action = f"📉 จุดเริ่มต้น: *เริ่ม Short* (~{len_pred} แท่ง)"
            position = dir_label
            entry_price = price_now
        # เพิ่มไม้กรณีถืออยู่แล้ว
        elif position == dir_label:
            action = f"➕ เพิ่มไม้ในเทรนด์ {dir_label.upper()} ต่อ (~{len_pred} แท่ง)"
        # สลับทิศทางเทรนด์ (จากถืออยู่ → ตรงข้าม)
        else:
            if position == 'up':
                pnl = (price_now - entry_price) / entry_price * 100
            else:
                pnl = (entry_price - price_now) / entry_price * 100
            action = f"🔁 ปิด {position.upper()} ที่กำไร {round(pnl, 2)}% → เริ่ม {dir_label.upper()} ใหม่"
            entry_price = price_now
            position = dir_label

    # แนวโน้ม sideway
    elif dir_label == 'sideway':
        if position is not None:
            action = f"⚠ แนวโน้ม Sideway → *ถือ {position.upper()} ต่อ* รอดูจังหวะใหม่"
        else:
            action = f"⏸ Sideway ~{len_pred} แท่ง → ยังไม่ควรเข้าเทรด"

    # ไม่เข้าเงื่อนไข
    else:
        if position is not None:
            # พิจารณาปิดสถานะหาก trend เปลี่ยนหรือความมั่นใจลด
            if adx_val < 25 or proba < 0.6:
                if position == 'up':
                    pnl = (price_now - entry_price) / entry_price * 100
                else:
                    pnl = (entry_price - price_now) / entry_price * 100
                action = f"🛑 สัญญาณอ่อน → ปิดสถานะ {position.upper()} ที่ {round(pnl, 2)}%"
                position = None
                entry_price = 0
            else:
                action = f"🔄 ถือ {position.upper()} ต่อ รอจุดออก"
        else:
            action = "❌ ยังไม่เข้าเงื่อนไขการเทรดที่ปลอดภัย"

    # print(
    #     f"🔮 แนวโน้มถัดไป: {dir_label.upper()} ต่อเนื่องประมาณ {len_pred} แท่ง")
    # print(f"🧠 ความมั่นใจ: {round(proba*100, 2)}% | ADX: {round(adx_val, 2)}")
    # print(f"🎯 คำแนะนำ: {action}")
    # print("=========================================")

    # Update the placeholder with the latest action
    placeholder.markdown(
        f"🔮 **แนวโน้มถัดไป:** {dir_label.upper()} ต่อเนื่องประมาณ {len_pred} แท่ง")
    placeholder.markdown(
        f"🧠 **ความมั่นใจ:** {round(proba*100, 2)}% | ADX: {round(adx_val, 2)}")
    placeholder.markdown(f"🎯 **คำแนะนำ:** {action}")
    placeholder.markdown("=========================================")

    # 建立文字訊息 TextMessage
    full_message = f"🔮 แนวโน้มถัดไป: {dir_label.upper()} ต่อเนื่องประมาณ {len_pred} แท่ง \n🧠 ความมั่นใจ: {round(proba*100, 2)}% | ADX: {round(adx_val, 2)}\n🎯 คำแนะนำ: {action}"
    text_message = TextMessage(text=full_message)
    push_message_request = PushMessageRequest(
        to=user_id, messages=[text_message])

    # 傳送訊息
    messaging_api.push_message(push_message_request=push_message_request)
    time.sleep(30)  # รอ 1 นาทีเพื่อให้ข้อมูลใหม่เข้ามา
