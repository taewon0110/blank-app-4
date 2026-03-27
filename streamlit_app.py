"""
세션 5-8: [프로젝트] AI + Streamlit 연동 (유가 예측 및 LLM 해석 대시보드)
실행: streamlit run streamlit_app.py

MBC-BLANK 스타일: 거대한 핵심 지표(RMSE 등) + 다크테마 시각화
기능 1: 사전 가중치 훈련 파일 없이 FRED/합성 기반 실시간 On-the-fly 학습 (선형회귀 / LSTM)
기능 2: 예측 결과를 바탕으로 HuggingFace Inference API 연동하여 정량 분석(Market Insight) 제공
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import platform
import os
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime

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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# ══════════════════════════════════════════════════════════════
#  앱 기본 설정
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="🛢️ AI Oil & Energy Dashboard", layout="wide", initial_sidebar_state="expanded")

# 프리미엄 다크 테마 CSS
premium_css = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 메트릭 카드 */
.metric-card {
    background: linear-gradient(135deg, rgba(30,30,50,0.9), rgba(20,20,40,0.95));
    border: 1px solid rgba(120,120,180,0.15);
    border-radius: 16px;
    padding: 24px 16px 18px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(100,100,200,0.15);
}
.metric-value {
    font-size: 2.8rem;
    font-weight: 200;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: rgba(255,255,255,0.45);
    margin-bottom: 4px;
}
.metric-sub {
    font-size: 0.8rem;
    color: rgba(255,255,255,0.35);
}

/* 가격 카드 */
.price-card {
    background: linear-gradient(135deg, rgba(20,20,45,0.95), rgba(15,15,35,0.98));
    border: 1px solid rgba(100,100,180,0.1);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.price-value {
    font-size: 1.6rem;
    font-weight: 300;
}
.price-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.4);
    margin-bottom: 4px;
}
.change-up { color: #2ecc71; }
.change-down { color: #e74c3c; }
.change-neutral { color: #95a5a6; }

/* AI 분석 결과 카드 */
.insight-card {
    background: linear-gradient(135deg, rgba(25,25,55,0.95), rgba(18,18,42,0.98));
    border-left: 3px solid #9b59b6;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 12px 0;
    line-height: 1.7;
    font-size: 0.95rem;
}

/* 사이드바 스타일 */
.sidebar-info {
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    padding: 12px;
    margin-top: 16px;
    font-size: 0.78rem;
    color: rgba(255,255,255,0.5);
    line-height: 1.6;
}
</style>
"""
st.markdown(premium_css, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  Hero Banner (서버 부하 0% - 클라이언트 브라우저 직접 로드)
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div style="width: 100%; max-height: 180px; overflow: hidden; border-radius: 16px; margin-bottom: 24px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); position: relative;">
    <img src="https://images.unsplash.com/photo-1613521140785-e85e427f8002?q=80&w=2000&auto=format&fit=crop" 
         style="width: 100%; object-fit: cover; object-position: center; filter: brightness(0.6) contrast(1.2);" 
         loading="lazy" alt="Market Data Background">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; width: 100%;">
        <h1 style="margin: 0; color: white; font-size: 2.5rem; letter-spacing: 2px; text-shadow: 0 2px 10px rgba(0,0,0,0.8);">AI MARKET INTELLIGENCE</h1>
        <p style="margin: 5px 0 0 0; color: #a5b1fa; font-size: 1rem; letter-spacing: 1px;">Real-time Predictive Analytics & Quantitative Insights</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  유가 데이터 로드 (FRED API - 완전 무료 & 차단 없음)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def load_oil_data(commodity, start, end):
    import pandas_datareader.data as web
    
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
        
    # Fallback: 합성 데이터 (비상용)
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

# 사이드바 하단 정보
st.sidebar.markdown(f"""
<div class="sidebar-info">
    <b>📡 Data Source</b>: FRED (Federal Reserve)<br>
    <b>🧠 Model</b>: {model_type} (On-the-fly)<br>
    <b>📊 Dataset</b>: {len(df):,} trading days<br>
    <b>🕐 Last Updated</b>: {datetime.now().strftime('%Y-%m-%d %H:%M')}
</div>
""", unsafe_allow_html=True)

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

# 추가 지표 계산
latest_price = actual_test[-1]
forecast_price = pred_future[-1]
price_change_pct = ((forecast_price - latest_price) / latest_price) * 100
direction = "UP" if price_change_pct > 0 else "DOWN" if price_change_pct < 0 else "FLAT"
direction_color = "change-up" if price_change_pct > 0 else "change-down" if price_change_pct < 0 else "change-neutral"
direction_emoji = "📈" if price_change_pct > 0 else "📉" if price_change_pct < 0 else "➡️"
r2_color = "#2ecc71" if r2 > 0.85 else "#f39c12" if r2 > 0.7 else "#e74c3c"

# ══════════════════════════════════════════════════════════════
#  UI 렌더링 (Premium Dark Mode)
# ══════════════════════════════════════════════════════════════

# 1) 가격 요약 카드 (최근가 / 예측가 / 변동률)
st.markdown("<br>", unsafe_allow_html=True)
p1, p2, p3 = st.columns(3)
p1.markdown(f"""
<div class="price-card">
    <div class="price-label">Latest Close</div>
    <div class="price-value" style="color: #2ecc71;">${latest_price:.2f}</div>
</div>
""", unsafe_allow_html=True)
p2.markdown(f"""
<div class="price-card">
    <div class="price-label">Forecast ({forecast_days}d)</div>
    <div class="price-value" style="color: #9b59b6;">${forecast_price:.2f}</div>
</div>
""", unsafe_allow_html=True)
p3.markdown(f"""
<div class="price-card">
    <div class="price-label">{direction_emoji} Expected Change</div>
    <div class="price-value {direction_color}">{price_change_pct:+.2f}%</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 2) 모델 성능 지표 카드
c1, c2, c3 = st.columns(3)
c1.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Root Mean Square Error</div>
    <div class="metric-value" style="color: #e8e8ff;">{rmse:.4f}</div>
    <div class="metric-sub">RMSE · lower is better</div>
</div>
""", unsafe_allow_html=True)
c2.markdown(f"""
<div class="metric-card">
    <div class="metric-label">R-Squared Score</div>
    <div class="metric-value" style="color: {r2_color};">{r2:.4f}</div>
    <div class="metric-sub">R² · closer to 1.0 is better</div>
</div>
""", unsafe_allow_html=True)
c3.markdown(f"""
<div class="metric-card">
    <div class="metric-label">Mean Absolute Error</div>
    <div class="metric-value" style="color: #e8e8ff;">{mae:.4f}</div>
    <div class="metric-sub">MAE · lower is better</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# 3) 차트
fig, ax = plt.subplots(figsize=(16, 7), facecolor='#0E1117')
ax.set_facecolor('#0E1117')

ax.plot(train_data.index, train_data["price"], label="Train", alpha=0.35, color="#a5b1fa", linewidth=1)
ax.plot(test_dates, actual_test, label="Actual (Test)", color="#2ecc71", linewidth=2.2)
ax.plot(test_dates, pred_test, label=f"Predicted ({model_type})", color="#e74c3c", linestyle="--", linewidth=1.5, alpha=0.85)
ax.plot(future_dates, pred_future, label=f"Forecast ({forecast_days}d)", color="#9b59b6", linewidth=3.5, marker="D", markersize=4, markerfacecolor="#c39bd3")

# 현재 가격 수평선
ax.axhline(y=latest_price, color="#f39c12", linestyle=":", alpha=0.4, linewidth=1)
ax.text(future_dates[-1], latest_price, f"  Current ${latest_price:.1f}", color="#f39c12", fontsize=9, alpha=0.6, va='center')

if show_confidence:
    std = np.std(actual_test - pred_test)
    ax.fill_between(future_dates, pred_future - 1.96*std, pred_future + 1.96*std, alpha=0.15, color="#9b59b6", label="95% CI")

# 예측 끝점 라벨
ax.annotate(f'${forecast_price:.1f}', xy=(future_dates[-1], forecast_price),
            xytext=(15, 10), textcoords='offset points',
            color='#c39bd3', fontsize=11, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#9b59b6', lw=1.5))

ax.set_title(f"{commodity} Price Forecast ({model_type} On-the-Fly)", fontsize=18, pad=15, color='white', fontweight='300')
ax.set_ylabel("Price (USD)", fontsize=12, color='white')
ax.tick_params(colors='white', labelsize=10)
ax.legend(fontsize=10, loc='upper left', facecolor='#0E1117', edgecolor='#333333', labelcolor='white', framealpha=0.9)
ax.grid(alpha=0.12, linestyle=':', color='#555555')
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('$%.0f'))

for spine in ax.spines.values():
    spine.set_color('#333333')

plt.tight_layout()
st.pyplot(fig)

# ══════════════════════════════════════════════════════════════
#  AI Insight (HuggingFace — 실시간 자동 연동 + 캐싱)
# ══════════════════════════════════════════════════════════════
HF_API_KEY = None
try:
    HF_API_KEY = st.secrets["HUGGINGFACE_API_KEY"]
except Exception:
    pass

@st.cache_data(show_spinner=False, ttl=1800)
def generate_ai_insight(api_key, _commodity, _model_type, _latest_price, _forecast_price, _change_pct, _rmse, _r2, _mae, _forecast_days, _n_data, _start, _end):
    """AI 분석 캐싱 함수 — 동일 파라미터면 캐시에서 즉시 로드 (TTL 30분)"""
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=api_key)
    response = client.chat_completion(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[
            {"role": "system", "content": "You are a top Wall Street quantitative analyst. Always respond in Korean. Be concise, sharp, and professional. Use financial terminology."},
            {"role": "user", "content": f"""다음은 {_model_type} 모델이 실시간으로 {_commodity} 자산의 가격을 예측한 데이터입니다:

- 최근 종가: ${_latest_price:.2f}
- {_forecast_days}일 뒤 예측 종가: ${_forecast_price:.2f}
- 예상 변동률: {_change_pct:+.2f}%
- 모델 평가: RMSE {_rmse:.4f}, R² {_r2:.4f}, MAE {_mae:.4f}
- 분석 기간: {_start} ~ {_end} ({_n_data}개 거래일)

이 지표들을 바탕으로, 해당 원자재의 미래 추세와 리스크를 분석하는 전문가 수준의 요약(Executive Summary)을 3문장 이내로 작성하세요. 반드시 시장 논리를 곁들이고, 냉정하고 날카로운 톤을 유지하세요."""}
        ],
        max_tokens=300,
        temperature=0.7,
    )
    return response.choices[0].message.content

if HF_API_KEY:
    st.markdown("### 🤖 AI Qualitative Market Insight")
    st.caption("📡 실시간 자동 분석 · 파라미터 변경 시 자동 갱신 · 30분 캐싱")
    
    # 수동 재분석 버튼
    col_btn, col_info = st.columns([1, 3])
    force_refresh = col_btn.button("🔄 재분석", use_container_width=True)
    if force_refresh:
        generate_ai_insight.clear()
    
    # 자동 실행 (캐싱)
    with st.spinner("🧠 AI가 시장 데이터를 분석하고 있습니다..."):
        try:
            insight = generate_ai_insight(
                HF_API_KEY, commodity, model_type,
                round(latest_price, 2), round(forecast_price, 2), round(price_change_pct, 2),
                round(rmse, 4), round(r2, 4), round(mae, 4),
                forecast_days, len(df), str(start_date), str(end_date)
            )
            st.markdown(f'<div class="insight-card">{insight}</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"API 호출 중 에러 발생: {e}")

# ══════════════════════════════════════════════════════════════
#  상세 데이터 테이블
# ══════════════════════════════════════════════════════════════
with st.expander("📋 Prediction Comparison Table"):
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
        }).background_gradient(subset=["Error %"], cmap="RdYlGn_r", vmin=-5, vmax=5),
        use_container_width=True
    )

# 예측 통계 요약
with st.expander("📊 Forecast Statistics"):
    fs1, fs2, fs3, fs4 = st.columns(4)
    fs1.metric("Min Forecast", f"${pred_future.min():.2f}")
    fs2.metric("Max Forecast", f"${pred_future.max():.2f}")
    fs3.metric("Avg Forecast", f"${pred_future.mean():.2f}")
    fs4.metric("Volatility (σ)", f"${np.std(pred_future):.2f}")
