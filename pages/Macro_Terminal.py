import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime

# ══════════════════════════════════════════════════════════════
#  앱 기본 설정
# ══════════════════════════════════════════════════════════════
st.set_page_config(page_title="🌐 Global Macro Terminal", layout="wide", initial_sidebar_state="expanded")

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
    font-size: 2.5rem;
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
    border-left: 3px solid #3498db; /* Blue for macro */
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
    <img src="https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?q=80&w=2000&auto=format&fit=crop" 
         style="width: 100%; object-fit: cover; object-position: center; filter: brightness(0.5) contrast(1.1);" 
         loading="lazy" alt="Global Macro Background">
    <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); text-align: center; width: 100%;">
        <h1 style="margin: 0; color: white; font-size: 2.5rem; letter-spacing: 2px; text-shadow: 0 2px 10px rgba(0,0,0,0.8);">GLOBAL MACRO TERMINAL</h1>
        <p style="margin: 5px 0 0 0; color: #81ecec; font-size: 1rem; letter-spacing: 1px;">Cross-Asset Market Data & AI Sentiment Analysis</p>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  데이터 파이프라인 (yfinance 캐싱)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=1800)
def fetch_macro_data(period="1y"):
    import pandas_datareader.data as web
    from datetime import datetime, timedelta
    
    # FRED Series IDs: 완전 무료 & 차단 우려 제로
    # SP500: S&P 500
    # DGS10: 10-Year Treasury Yield
    # GOLDAMGBNP: London Gold Fixing
    # DTWEXBGS: Trade Weighted U.S. Dollar Index
    
    tickers = {
        "S&P 500 (Equity)": "SP500",
        "US 10-Yr Yield (Rates)": "DGS10",
        "Gold (Safe Haven)": "GOLDAMGBNP",
        "USD Index (Currency)": "DTWEXBGS"
    }
    
    end = datetime.now()
    if period == "1mo": start = end - timedelta(days=30)
    elif period == "3mo": start = end - timedelta(days=90)
    elif period == "6mo": start = end - timedelta(days=180)
    elif period == "1y": start = end - timedelta(days=365)
    elif period == "2y": start = end - timedelta(days=730)
    else: start = end - timedelta(days=1825)

    data = {}
    for name, ticker in tickers.items():
        try:
            # FRED API 호출
            df = web.DataReader(ticker, 'fred', start, end)
            if df is not None and not df.empty:
                # FRED 데이터는 Ticker명과 컬럼명이 동일함
                data[name] = df[ticker]
        except Exception as e:
            # 개별 데이터 실패 시 skip
            continue
            
    if data:
        combined = pd.DataFrame(data)
        # 결측값 채우기 (시장 전반의 휴장일 등 처리)
        combined.ffill(inplace=True)
        combined.bfill(inplace=True)
        # 시간 정보만 사용
        combined.index = pd.to_datetime(combined.index).date
        combined.index.name = "Date"
        return combined
        
    return pd.DataFrame()

# ══════════════════════════════════════════════════════════════
#  AI 파이프라인 (HuggingFace Serverless API - cached)
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=1800)
def generate_macro_insight(api_key, sp500_ret, yield_val, gold_ret, usd_ret):
    from huggingface_hub import InferenceClient
    client = InferenceClient(token=api_key)
    try:
        response = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[
                {"role": "system", "content": "You are a top Wall Street Chief Economist. Always respond in Korean. Be concise, sharp, and professional. Synthesize cross-asset dynamics."},
                {"role": "user", "content": f"""다음은 현재 글로벌 매크로 자산 지표입니다 (최근 1년 변동률/현재값):

- S&P 500 (주식): {sp500_ret:+.2f}%
- 미국 10년물 국채 금리: {yield_val:.2f}%
- 금 가격 (안전자산): {gold_ret:+.2f}%
- 달러 인덱스: {usd_ret:+.2f}%

이 교차 자산(Cross-Asset) 시장 데이터를 종합하여, 현재 시장 상황이 Risk-On(위험 선호)인지 Risk-Off(안전 자산 선호)인지, 유동성과 경기 침체 리스크는 어떤 상태인지 3문장 이내 전문가 수준(Executive Summary)으로 작성하세요."""}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"AI 연결 실패: {str(e)[:50]}"

# ══════════════════════════════════════════════════════════════
#  사이드바 설정
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### ⚙️ Terminal Settings")
    period_select = st.selectbox("View Horizon", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)
    
    st.markdown("""
        <div class="sidebar-info">
            🚀 <strong>Data Feed:</strong> Yahoo Finance<br>
            🧠 <strong>Analysis:</strong> Qwen2.5-7B (HF API)<br>
            ⏳ <strong>Update Freq:</strong> Sub-second (Cached)<br>
            🛡️ <strong>Latency:</strong> Zero-overhead
        </div>
    """, unsafe_allow_html=True)
    
    try:
        from dotenv import load_dotenv
        import os
        load_dotenv("c:/Users/USER/Desktop/20260325/.env")
        HF_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")
        if not HF_API_KEY:
            HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", None)
    except:
        HF_API_KEY = st.secrets.get("HUGGINGFACE_API_KEY", None)

# ══════════════════════════════════════════════════════════════
#  메인 UI 렌더링
# ══════════════════════════════════════════════════════════════
st.markdown("### 📊 Cross-Asset Matrix")

df = fetch_macro_data(period=period_select)

if df.empty:
    st.error("데이터 로드에 실패했습니다. API 연결 상태를 확인해주세요.")
    st.stop()

# 최신 지표 및 변동 계산 로직
latest = df.iloc[-1]
last_year = df.iloc[0] # The requested period's first day

sp500_latest = latest.get("S&P 500 (Equity)", 0)
sp500_ret = ((sp500_latest - last_year.get("S&P 500 (Equity)", 1)) / last_year.get("S&P 500 (Equity)", 1)) * 100

tnx_latest = latest.get("US 10-Yr Yield (Rates)", 0)
tnx_ret = tnx_latest - last_year.get("US 10-Yr Yield (Rates)", 0) # bps change roughly or absolute % change

gold_latest = latest.get("Gold (Safe Haven)", 0)
gold_ret = ((gold_latest - last_year.get("Gold (Safe Haven)", 1)) / last_year.get("Gold (Safe Haven)", 1)) * 100

usd_latest = latest.get("USD Index (Currency)", 0)
usd_ret = ((usd_latest - last_year.get("USD Index (Currency)", 1)) / last_year.get("USD Index (Currency)", 1)) * 100

def get_color_class(val):
    if val > 0: return "change-up"
    elif val < 0: return "change-down"
    return "change-neutral"

def format_change(val, is_yield=False):
    symbol = "+" if val > 0 else ""
    suffix = " bps" if is_yield else "%"
    display_val = val * 100 if is_yield else val # TNX absolute diff approximation
    return f"{symbol}{display_val:.2f}{suffix}"

cols = st.columns(4)

with cols[0]:
    val_cls = get_color_class(sp500_ret)
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">S&P 500 🇺🇸</div>
        <div class="price-value">{sp500_latest:,.2f}</div>
        <div class="metric-sub {val_cls}">{format_change(sp500_ret)}</div>
    </div>
    """, unsafe_allow_html=True)

with cols[1]:
    # For yields, rising yields is bad for bonds, but we leave color neutral or specific logic. 
    val_cls = get_color_class(tnx_ret)
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">US 10-Yr Yield 📈</div>
        <div class="price-value">{tnx_latest:.3f}%</div>
        <div class="metric-sub {val_cls}">{format_change(tnx_ret, True)}</div>
    </div>
    """, unsafe_allow_html=True)

with cols[2]:
    val_cls = get_color_class(gold_ret)
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Gold 🥇</div>
        <div class="price-value">${gold_latest:,.2f}</div>
        <div class="metric-sub {val_cls}">{format_change(gold_ret)}</div>
    </div>
    """, unsafe_allow_html=True)

with cols[3]:
    val_cls = get_color_class(usd_ret)
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">USD Index 💵</div>
        <div class="price-value">{usd_latest:.2f}</div>
        <div class="metric-sub {val_cls}">{format_change(usd_ret)}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  시그널 차트 (정규화된 비교 차트)
# ══════════════════════════════════════════════════════════════
st.markdown("### 📈 Normalized Performance Matrix")
# 시작점을 100으로 정규화하여 4개 지표를 한 차트에서 비교
df_norm = (df / df.iloc[0]) * 100
st.line_chart(df_norm, use_container_width=True, height=350)

# ══════════════════════════════════════════════════════════════
#  AI 인사이트
# ══════════════════════════════════════════════════════════════
st.markdown("### 🌐 AI Chief Economist Insight")
col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 재분석", use_container_width=True):
        generate_macro_insight.clear()

if HF_API_KEY:
    insight_text = generate_macro_insight(HF_API_KEY, sp500_ret, tnx_latest, gold_ret, usd_ret)
    st.markdown(f"""
    <div class="insight-card">
        {insight_text}
    </div>
    """, unsafe_allow_html=True)
else:
    st.warning("HuggingFace API Key가 필요합니다.")
