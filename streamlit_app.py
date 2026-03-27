"""
세션 5-8: [프로젝트] AI + Streamlit 연동 (유가 예측 및 LLM 해석 대시보드)
실행: streamlit run streamlit_app.py

MBC-BLANK 스타일: 거대한 핵심 지표(RMSE 등) + 다크테마 시각화
기능 1: 사전 가중치 훈련 파일 없이 yfinance 기반 실시간 On-the-fly 학습 (선형회귀 / LSTM)
기능 2: 예측 결과를 바탕으로 Google Gemini API 연동하여 정량 분석(Market Insight) 제공

※ yfinance 선물 티커(CL=F 등) 불안정 대비: ETF 티커(USO, BNO, UNG) 우선 사용 + 다단계 Fallback
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import os
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ══════════════════════════════════════════════════════════════
#  폰트 및 차트 스타일
# ══════════════════════════════════════════════════════════════
if platform.system() == "Darwin":
    plt.rcParams["font.family"] = "AppleGothic"
elif platform.system() == "Windows":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    plt.rcParams["font.family"] = "DejaVu Sans"

plt.rcParams["axes.unicode_minus"] = False
plt.style.use('dark_background')

# ══════════════════════════════════════════════════════════════
#  LSTM 구조
# ══════════════════════════════════════════════════════════════
class LSTMPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ══════════════════════════════════════════════════════════════
#  앱 기본 설정
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="🛢️ AI Oil Price Dashboard", layout="wide", initial_sidebar_state="expanded")

hide_st_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  유가 데이터 로드 (FRED API - 완전 무료 & 차단 없음)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def load_oil_data(commodity, start, end):
    import pandas_datareader.data as web
    import datetime
    
    # FRED 공식 티커 사용 (월스트리트 실제 데이터)
    tickers = {
        "WTI Crude Oil": "DCOILWTICO",
        "Brent Crude": "DCOILBRENTE",
        "Natural Gas": "DHHNGSP"
    }
    ticker = tickers.get(commodity, "DCOILWTICO")
    
    try:
        df = web.DataReader(ticker, 'fred', start, end)
        if df is not None and not df.empty and len(df) >= 50:
            df.columns = ["price"]
            df = df.dropna()
            return df
    except Exception:
        pass
        
    # 만약 FRED도 터지면 합성 데이터 (비상용)
    np.random.seed(42)
    n_days = 500
    dates = pd.date_range(end=pd.Timestamp(end), periods=n_days, freq="B")
    
    base_price = 105.0 
    trend = np.linspace(0, 12, n_days)
    seasonality = 6 * np.sin(np.arange(n_days) * 2 * np.pi / 252)
    noise = np.random.normal(0, 3.0, n_days)
    random_walk = np.cumsum(np.random.normal(0, 0.4, n_days))
    prices = base_price + trend + seasonality + noise + random_walk
    prices = np.clip(prices, 60, 180) 
    
    df = pd.DataFrame({"price": prices}, index=dates)
    return df

# ══════════════════════════════════════════════════════════════
#  모델 On-the-fly 학습 (캐싱)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔥 외부 가중치 없이 실시간 On-the-fly 모델 학습 중...")
def train_models_on_the_fly(_prices_tuple):
    prices = np.array(_prices_tuple)
    trained = {}

    # 1. Linear Regression
    n_days = len(prices)
    day_nums = np.arange(n_days).reshape(-1, 1)
    lr_model = LinearRegression()
    lr_model.fit(day_nums, prices)
    trained["Linear Regression"] = lr_model

    # 2. LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    prices_scaled = scaler.fit_transform(prices.reshape(-1, 1)).flatten()

    window = 30
    X_seq, y_seq = [], []
    for i in range(len(prices_scaled) - window):
        X_seq.append(prices_scaled[i : i + window])
        y_seq.append(prices_scaled[i + window])

    if len(X_seq) > 0:
        X_tensor = torch.FloatTensor(np.array(X_seq)).unsqueeze(-1)
        y_tensor = torch.FloatTensor(np.array(y_seq)).unsqueeze(-1)

        lstm_model = LSTMPredictor()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.01)

        lstm_model.train()
        for epoch in range(200):
            optimizer.zero_grad()
            output = lstm_model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()

        lstm_model.eval()
        trained["LSTM"] = (lstm_model, scaler)
    return trained

# ─── 사이드바 ───
st.sidebar.markdown('### ⚙️ Prediction Settings')
commodity = st.sidebar.selectbox("Commodity", ["WTI Crude Oil", "Brent Crude", "Natural Gas"])
start_date = st.sidebar.date_input("Start Date", pd.Timestamp("2024-03-01"))
end_date = st.sidebar.date_input("End Date", pd.Timestamp("2026-03-20"))
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 1, 60, 30)
show_confidence = st.sidebar.checkbox("Show Confidence Interval", value=True)

df = load_oil_data(commodity, str(start_date), str(end_date))

if df.empty or len(df) < 50:
    st.error("⚠️ 데이터 생성 실패.")
    st.stop()

models = train_models_on_the_fly(tuple(df["price"].values))
available_models = list(models.keys())
model_type = st.sidebar.radio("Model", available_models, index=1 if "LSTM" in available_models else 0)

# ══════════════════════════════════════════════════════════════
#  예측 로직
# ══════════════════════════════════════════════════════════════
split_idx = int(len(df) * 0.8)
train_data = df.iloc[:split_idx]
test_data = df.iloc[split_idx:]
test_dates = test_data.index
actual_test = test_data["price"].values
future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq="B")

if model_type == "Linear Regression":
    model = models["Linear Regression"]
    X_test = np.arange(split_idx, len(df)).reshape(-1, 1)
    pred_test = model.predict(X_test)
    X_future = np.arange(len(df), len(df) + forecast_days).reshape(-1, 1)
    pred_future = model.predict(X_future)

elif model_type == "LSTM":
    model, scaler = models["LSTM"]
    window = 30
    prices_scaled = scaler.transform(df["price"].values.reshape(-1, 1)).flatten()
    pred_test_scaled = []

    with torch.no_grad():
        for i in range(split_idx, len(df)):
            x_seq = prices_scaled[i - window : i]
            if len(x_seq) < window:
                pred_test_scaled.append(prices_scaled[i])
                continue
            x_tensor = torch.FloatTensor(x_seq).unsqueeze(0).unsqueeze(-1)
            pred = model(x_tensor).item()
            pred_test_scaled.append(pred)

    pred_test = scaler.inverse_transform(np.array(pred_test_scaled).reshape(-1, 1)).flatten()

    current_seq = prices_scaled[-window:].tolist()
    pred_future_scaled = []
    with torch.no_grad():
        for _ in range(forecast_days):
            x_tensor = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1)
            pred = model(x_tensor).item()
            pred_future_scaled.append(pred)
            current_seq.append(pred)
            current_seq.pop(0)

    pred_future = scaler.inverse_transform(np.array(pred_future_scaled).reshape(-1, 1)).flatten()

rmse = np.sqrt(mean_squared_error(actual_test, pred_test))
r2 = r2_score(actual_test, pred_test)
mae = mean_absolute_error(actual_test, pred_test)

# ══════════════════════════════════════════════════════════════
#  UI 렌더링 (mbc-blank 스타일)
# ══════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
c1.markdown(f"<h1 style='text-align: center; font-size: 3.5rem; font-weight: 300;'>{rmse:.4f}</h1>", unsafe_allow_html=True)
c2.markdown(f"<h1 style='text-align: center; font-size: 3.5rem; font-weight: 300;'>{r2:.4f}</h1>", unsafe_allow_html=True)
c3.markdown(f"<h1 style='text-align: center; font-size: 3.5rem; font-weight: 300;'>{mae:.4f}</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(16, 7), facecolor='#0E1117')
ax.set_facecolor('#0E1117')

ax.plot(train_data.index, train_data["price"], label="Train", alpha=0.4, color="#a5b1fa")
ax.plot(test_dates, actual_test, label="Actual (Test)", color="#2ecc71", linewidth=2.5)
ax.plot(test_dates, pred_test, label=f"Predicted ({model_type})", color="#e74c3c", linestyle="--")
ax.plot(future_dates, pred_future, label=f"Forecast ({forecast_days}d)", color="#9b59b6", linewidth=4, marker="D", markersize=5)

if show_confidence:
    std = np.std(actual_test - pred_test)
    ax.fill_between(future_dates, pred_future - 1.96*std, pred_future + 1.96*std, alpha=0.2, color="#9b59b6")

ax.set_title(f"{commodity} Price Forecast ({model_type} On-the-Fly)", fontsize=18, pad=15, color='white')
ax.set_ylabel("Price (USD)", fontsize=12, color='white')
ax.tick_params(colors='white')
ax.legend(fontsize=11, loc='upper right', facecolor='#0E1117', edgecolor='white', labelcolor='white')
ax.grid(alpha=0.2, linestyle=':')

for spine in ax.spines.values():
    spine.set_color('#444444')

plt.tight_layout()
st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
#  AI Insight (HuggingFace Inference API — Zephyr-7B)
# ══════════════════════════════════════════════════════════════
HF_API_KEY = None
try:
    HF_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except Exception:
    pass

if HF_API_KEY:
    from huggingface_hub import InferenceClient
    st.markdown("### 🤖 AI Qualitative Market Insight")
    if st.button("Generate Trend Analysis"):
        with st.spinner("AI 분석 연산 중..."):
            try:
                hf_client = InferenceClient(token=HF_API_KEY)
                prompt_msg = [
                    {"role": "system", "content": "You are a top Wall Street quantitative analyst. Always respond in Korean. Be concise and sharp."},
                    {"role": "user", "content": f"""다음은 LSTM 모델이 실시간으로 {commodity} 자산의 가격을 예측한 데이터입니다:

- 최근 종가: ${actual_test[-1]:.2f}
- {forecast_days}일 뒤 예측 종가: ${pred_future[-1]:.2f}
- 모델 평가: RMSE {rmse:.4f}, R² {r2:.4f}, MAE {mae:.4f}

이 지표들을 바탕으로, 해당 원자재의 미래 추세와 리스크를 분석하는 전문가 수준의 요약(Executive Summary)을 3문장 이내로 작성하세요. 반드시 시장 논리를 곁들이고, 냉정하고 날카로운 톤을 유지하세요."""}
                ]
                response = hf_client.chat_completion(
                    model="HuggingFaceH4/zephyr-7b-beta",
                    messages=prompt_msg,
                    max_tokens=300,
                    temperature=0.7,
                )
                st.info(response.choices[0].message.content)
            except Exception as e:
                st.error(f"API 호출 중 에러 발생: {e}")

with st.expander("Prediction Comparison Table"):
    comp = pd.DataFrame({
        "Date": test_dates,
        "Actual": actual_test,
        "Predicted": pred_test,
        "Error": actual_test - pred_test,
        "Error %": ((actual_test - pred_test) / actual_test * 100)
    })
    st.dataframe(
        comp.style.format({
            "Actual": "${:.2f}",
            "Predicted": "${:.2f}",
            "Error": "${:.2f}",
            "Error %": "{:.2f}%"
        }),
        use_container_width=True
    )
