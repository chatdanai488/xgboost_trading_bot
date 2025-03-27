import os
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
import streamlit as st
import time

# ===== เตรียมเว็บ UI =====
st.set_page_config(page_title="📈 Forex RL Dashboard", layout="centered")
st.title("🤖 สัญญาณเทรดแบบ RL + LightGBM")
signal_placeholder = st.empty()
price_placeholder = st.empty()

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


# สร้าง label ว่าราคาจะขึ้นหรือไม่ในอีก 3 แท่งเทียน
df['Target'] = (df['Close'].shift(-3) > df['Close']).astype(int)

# ลบค่า NaN
df.dropna(inplace=True)


features = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'RSI', 'EMA_fast', 'EMA_slow',
    'BollingerBands', 'ADX', 'OBV',
    'Volume_Change', 'Candle_Body', 'Range', 'Return'
]

X = df[features]
y = df['Target']

train_size = int(0.8 * len(df))
X_train, y_train = X[:train_size], y[:train_size]

model = lgb.LGBMClassifier()
model.fit(X_train, y_train)

# ทำนายความน่าจะเป็นว่า "จะขึ้น"
df['LGB_Predict'] = model.predict_proba(X)[:, 1]


class ForexEnv(gym.Env):
    def __init__(self, df):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.action_space = spaces.Discrete(3)  # 0: Hold, 1: Buy, 2: Sell

        # Initialize observation space dynamically
        dummy_obs = self._get_obs(0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32)

        self.trades = []
        self.reset()

    def reset(self):
        self.i = 0
        self.balance = 0
        self.position = None
        self.entry_price = 0
        self.position_entry_step = None
        self.trades = []
        return self._get_obs()

    def _get_obs(self, index=None):
        i = self.i if index is None else index
        row = self.df.iloc[i]

        # ป้องกันหารด้วย 0
        candle_range = row['Range'] if row['Range'] != 0 else 1e-6

        return np.array([
            # สมมติ BTC/USD มีระดับราคาหลักหมื่น
            row['Open'] / 100000,
            row['High'] / 100000,
            row['Low'] / 100000,
            row['Close'] / 100000,
            row['Volume'] / 1e6,                          # Normalize volume
            row['RSI'] / 100,
            row['EMA_fast'] / row['EMA_slow'],
            # เพิ่มความสัมพันธ์กับราคา
            row['EMA_slow'] / row['Close'],
            # lower band normalized
            row['BollingerBands'] / row['Close'],
            row['ADX'] / 100,
            row['OBV'] / 1e6,
            row['Volume_Change'],                         # อัตราการเปลี่ยนแปลง
            row['Candle_Body'] / candle_range,
            candle_range / row['Close'],                  # Range normalized
            row['Return'],                                 # ราคาเปลี่ยนแปลง %
            row['LGB_Predict']  # ← เพิ่มตรงนี้
        ], dtype=np.float32)

    def step(self, action):
        reward = 0
        row = self.df.iloc[self.i]
        price_now = row['Close']
        action_label = ['HOLD', 'BUY', 'SELL'][action]

        realized = None  # สำหรับเก็บกำไรจริงเวลา close position

        if action == 1:  # BUY
            if self.position is None:
                self.entry_price = price_now
                self.position = 'long'
                self.position_entry_step = self.i
            else:
                reward = -10  # Penalty for invalid action

        elif action == 2:  # SELL
            if self.position == 'long':
                reward = price_now - self.entry_price
                realized = reward
                self.balance += reward
                self.position = None
                self.entry_price = 0
                self.position_entry_step = None
            else:
                reward = -10  # Penalty for invalid sell

        # Log รายการเทรด
        self.trades.append({
            'step': self.i,
            'price': price_now,
            'action': action_label,
            'reward': reward,
            'balance': self.balance,
            'position': self.position if self.position else 'none',
            'realized': realized
        })

        self.i += 1
        done = self.i >= len(self.df) - 2
        return self._get_obs(), reward, done, {}

    def render(self):
        print(
            f"Step {self.i}, Balance: {self.balance:.2f}, Position: {self.position}")

    def get_trades_log(self):
        return pd.DataFrame(self.trades)

    def get_trade_metrics(self):
        df = pd.DataFrame(self.trades)

        # เฉพาะ trade ที่มีกำไรขาดทุนจริง
        closed_trades = df[df['realized'].notnull()]

        if closed_trades.empty:
            return {
                "win_rate": 0,
                "avg_profit": 0,
                "max_drawdown": 0
            }

        profits = closed_trades['realized']
        win_rate = (profits > 0).sum() / len(profits)
        avg_profit = profits.mean()

        # Max drawdown จาก balance history
        balances = df['balance'].cummax() - df['balance']
        max_drawdown = balances.max()

        return {
            "win_rate": round(win_rate * 100, 2),
            "avg_profit": round(avg_profit, 2),
            "max_drawdown": round(max_drawdown, 2)
        }


model_path = "ppo_forex_model.zip"

env = ForexEnv(df)

if os.path.exists(model_path):
    print("📂 พบโมเดลเก่า → โหลดเพื่อฝึกต่อ")
    model_rl = PPO.load("ppo_forex_model", env=env)
else:
    print("🆕 ยังไม่มีโมเดล → สร้างใหม่จาก MlpPolicy")
    model_rl = PPO("MlpPolicy", env, verbose=1)

# ฝึก
model_rl.learn(total_timesteps=10000)

# บันทึก
model_rl.save("ppo_forex_model")


obs = env.reset()
done = False
while not done:
    action, _ = model_rl.predict(obs)
    obs, reward, done, _ = env.step(action)

env.render()


# log และสถิติ
trades_df = env.get_trades_log()
print(trades_df)

metrics = env.get_trade_metrics()
print(f"📈 Win Rate: {metrics['win_rate']}%")
print(f"💰 Avg Profit: {metrics['avg_profit']}")
print(f"📉 Max Drawdown: {metrics['max_drawdown']}")


# หรือ export เป็น CSV
trades_df.to_csv("trading_log.csv", index=False)


def decide_trade_action(df, model):
    env = ForexEnv(df)
    obs = env._get_obs(index=len(df) - 1).reshape(1, -1)
    action, _ = model.predict(obs)
    return ['HOLD', 'BUY', 'SELL'][action[0]]


signal = decide_trade_action(df, model_rl)
print(f"✅ สัญญาณล่าสุดจากโมเดล: {signal}")


# ===== แสดงผลแบบ Real-Time =====
while True:
    try:

        signal = decide_trade_action(df)
        price = df.iloc[-1]['Close']

        signal_placeholder.markdown(f"### 📢 สัญญาณล่าสุด: **{signal}**")
        price_placeholder.metric("ราคาปัจจุบัน BTC/USDT", f"${price:.2f}")

        time.sleep(10)
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาด: {e}")
        time.sleep(5)
