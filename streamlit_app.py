"""
세션 5-8: [프로젝트] AI + Streamlit 연동 (유가 예측 및 LLM 해석 대시보드)
실행: streamlit run streamlit_app.py

MBC-BLANK 스타일: 거대한 핵심 지표(RMSE 등) + 다크테마 시각화
기능 1: 사전 가중치 훈련 파일 없이 yfinance 기반 실시간 On-the-fly 학습 (선형회귀 / LSTM)
기능 2: 예측 결과를 바탕으로 Anthropic (Claude) API 연동하여 정량 분석(Market Insight) 제공
"""
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import platform
import os
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dotenv import load_dotenv
import anthropic

# ══════════════════════════════════════════════════════════════
#  환경 변수 로드 (로컬 .env 또는 Streamlit Secrets)
# ══════════════════════════════════════════════════════════════
# 상위 디렉토리의 .env 파일 로드 시도 (로컬용)
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=env_path)

API_KEY = os.environ.get("ANTHROPIC_API_KEY")
if not API_KEY:
    try:
        API_KEY = st.secrets["ANTHROPIC_API_KEY"]
    except:
        API_KEY = None

# ══════════════════════════════════════════════════════════════
#  스타일 및 폰트 설정
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
st.set_page_config(page_title="🛢️ AI Oil & Market Dashboard", layout="wide", initial_sidebar_state="expanded")

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  데이터 파이프라인 (캐싱)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_oil_data(commodity, start, end):
    tickers = {"WTI Crude Oil": "CL=F", "Brent Crude": "BZ=F", "Natural Gas": "NG=F"}
    ticker = tickers.get(commodity, "CL=F")
    
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty: return pd.DataFrame()
    
    if isinstance(df.columns, pd.MultiIndex):
        df = df['Close']
    else:
        df = df[['Close']]
        
    df.columns = ["price"]
    return df.dropna()

@st.cache_resource(show_spinner="🔥 외부 가중치 없이 실시간 On-the-fly 모델 학습 중...")
def train_models_on_the_fly(prices):
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
        for epoch in range(60): # 빠른 실시간 학습을 위한 에포크
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
start_date = st.sidebar.date_input("Start Date", pd.Timestamp.today() - pd.Timedelta(days=700))
end_date = st.sidebar.date_input("End Date", pd.Timestamp.today())
forecast_days = st.sidebar.slider("Forecast Horizon (days)", 1, 60, 30)
show_confidence = st.sidebar.checkbox("Show Confidence Interval", value=True)

df = load_oil_data(commodity, str(start_date), str(end_date))
if df.empty or len(df) < 50:
    st.error("데이터 로드 실패 또는 기간이 너무 짧습니다.")
    st.stop()

models = train_models_on_the_fly(df["price"].values)
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

    # 미래 예측
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
c1.markdown(f"<h1 style='text-align: center; color: white; font-size: 3.5rem; font-weight: 300;'>{rmse:.4f}</h1>", unsafe_allow_html=True)
c2.markdown(f"<h1 style='text-align: center; color: white; font-size: 3.5rem; font-weight: 300;'>{r2:.4f}</h1>", unsafe_allow_html=True)
c3.markdown(f"<h1 style='text-align: center; color: white; font-size: 3.5rem; font-weight: 300;'>{mae:.4f}</h1>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

fig, ax = plt.subplots(figsize=(16, 7), facecolor='black')
ax.set_facecolor('black')

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
ax.legend(fontsize=11, loc='upper right', facecolor='black', edgecolor='white', labelcolor='white')
ax.grid(alpha=0.2, linestyle=':')

for spine in ax.spines.values():
    spine.set_color('#444444')

plt.tight_layout()
st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
#  AI Insight (LLM 연동)
# ══════════════════════════════════════════════════════════════
if API_KEY:
    st.markdown("### 🤖 Claude 3.5 Qualitative Market Insight")
    if st.button("Generate Trend Analysis"):
        with st.spinner("Claude API 연산 중..."):
            try:
                client = anthropic.Anthropic(api_key=API_KEY)
                prompt = f"""
                당신은 월스트리트 최고 수준의 정량(Quant) 분석 AI입니다.
                다음은 LSTM 모델이 실시간으로 {commodity} 자산의 가격을 예측한 데이터입니다:
                
                - 최근 종가: ${actual_test[-1]:.2f}
                - {forecast_days}일 뒤 예측 종가:: ${pred_future[-1]:.2f}
                - 모델 평가: RMSE {rmse:.4f}, R-Squared {r2:.4f}, MAE {mae:.4f}
                
                이 지표들을 바탕으로, 해당 원자재의 미래 추세와 리스크를 분석하는 전문가 수준의 요약(Executive Summary)을 3문장 이내로 작성하세요. 반드시 시장 논리를 곁들이고, 냉정하고 날카로운 톤을 유지하세요.
                """
                response = client.messages.create(
                    model="claude-3-5-sonnet-20240620",
                    max_tokens=250,
                    messages=[{"role": "user", "content": prompt}]
                )
                
                st.info(response.content[0].text)
            except Exception as e:
                st.error(f"API 호출 중 에러 발생: {e}")
else:
    st.warning("Anthropic API 키가 설정되지 않아 AI Market Insight 기능을 사용할 수 없습니다. (.env 또는 st.secrets 확인)")
    
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
