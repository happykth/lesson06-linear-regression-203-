import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)
LAST_YEAR = 2025      # 수업의 기준 기간 끝
MIN_DAYS = 300        # 관측일이 이보다 적은 해는 제외
RECENT_YEARS = 20     # '최근' 기울기를 구할 때 쓰는 연수

st.set_page_config(page_title="기온 예측기", page_icon="🌡️", layout="centered")


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_annual_mean() -> pd.DataFrame:
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]

    df["날짜"] = pd.to_datetime(df["날짜"].astype(str).str.strip(), errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df = df.dropna(subset=["날짜", "평균기온"])  # 관측값이 있는 날만 사용

    df["연도"] = df["날짜"].dt.year
    df = df[df["연도"] <= LAST_YEAR]  # 기준 기간 이후 데이터 제외

    annual = (
        df.groupby("연도")["평균기온"]
        .agg(연평균기온="mean", 관측일수="count")
        .reset_index()
    )
    annual = annual[annual["관측일수"] >= MIN_DAYS]  # 관측일 부족한 해 제외
    return annual.reset_index(drop=True)


st.title("🌡️ 기온 예측기")
st.caption("서울의 연평균기온에 직선(선형 회귀)을 맞추고, 고른 연도의 기온을 예측합니다.")

try:
    annual = load_annual_mean()
except Exception as e:
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

if len(annual) < 2:
    st.error("회귀 직선을 만들기에 데이터가 부족해요.")
    st.stop()

x = annual["연도"].to_numpy(dtype=float)
y = annual["연평균기온"].to_numpy(dtype=float)



def fit_line(xs: np.ndarray, ys: np.ndarray):
    """직선 기울기, 절편, 상관계수, 기울기의 표준오차를 구한다."""
    s, b = np.polyfit(xs, ys, 1)
    rr = float(np.corrcoef(xs, ys)[0, 1])
    resid = ys - (s * xs + b)
    se = float(np.sqrt(np.sum(resid**2) / (len(xs) - 2) / np.sum((xs - xs.mean()) ** 2)))
    return float(s), float(b), rr, se


n_years = len(annual)
start_year = int(annual["연도"].min())
end_year = int(annual["연도"].max())

# 전체 기간으로 구한 직선 (예측에 사용)
slope, intercept, r, slope_se = fit_line(x, y)

# 최근 N년만으로 구한 직선 (기울기 비교용)
recent_mask = x > end_year - RECENT_YEARS
x_recent, y_recent = x[recent_mask], y[recent_mask]
recent_start = int(x_recent.min())
n_recent = len(x_recent)
slope_r, intercept_r, r_recent, slope_r_se = fit_line(x_recent, y_recent)

# --- 직선을 만든 데이터 정보 ---
c1, c2, c3 = st.columns(3)
c1.metric("직선을 만든 해의 개수", f"{n_years}개")
c2.metric("시작 연도", f"{start_year}년")
c3.metric("끝 연도", f"{end_year}년")
st.caption(
    f"{LAST_YEAR}년까지의 데이터만 사용했고, 관측일이 {MIN_DAYS}일 미만인 해는 제외했습니다."
)

# --- 연도 선택 & 예측 ---
year = st.slider("예측할 연도를 고르세요", min_value=1900, max_value=2100, value=2050, step=1)
pred = slope * year + intercept

st.markdown(
    f"""
    <div style="text-align:center; padding:1.2rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">{year}년 서울의 예상 연평균기온</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.2;">{pred:.2f} °C</div>
    </div>
    """,
    unsafe_allow_html=True,
)
if year < start_year or year > end_year:
    st.info(
        f"{year}년은 데이터 범위({start_year}~{end_year}년) 밖이라, 직선을 연장해 얻은 값이에요. "
        "범위에서 멀어질수록 믿기 어려워집니다."
    )

# --- 산점도 + 회귀 직선 ---
line_x = np.array([min(start_year, year), max(end_year, year)], dtype=float)
line_y = slope * line_x + intercept

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=annual["연도"],
        y=annual["연평균기온"],
        mode="markers",
        name="연평균기온",
        marker=dict(size=8, opacity=0.8),
        hovertemplate="%{x}년<br>%{y:.2f} °C<extra></extra>",
    )
)
fig.add_trace(
    go.Scatter(
        x=line_x,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3, color="crimson"),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=[recent_start, end_year],
        y=[slope_r * recent_start + intercept_r, slope_r * end_year + intercept_r],
        mode="lines",
        name=f"최근 {n_recent}년 직선",
        line=dict(width=3, color="royalblue", dash="dash"),
        hoverinfo="skip",
    )
)
fig.add_trace(
    go.Scatter(
        x=[year],
        y=[pred],
        mode="markers",
        name=f"{year}년 예측",
        marker=dict(size=16, symbol="star", color="gold", line=dict(width=1.5, color="black")),
        hovertemplate=f"{year}년 예측<br>{pred:.2f} °C<extra></extra>",
    )
)
fig.update_layout(
    xaxis_title="연도",
    yaxis_title="연평균기온 (°C)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    margin=dict(l=10, r=10, t=40, b=10),
    height=480,
)
st.plotly_chart(fig, width="stretch")

# --- 상관계수 & 직선 식 ---
st.metric("상관계수 (r)", f"{r:.3f}")
sign = "+" if intercept >= 0 else "-"
st.markdown(f"**회귀 직선:** 기온 = {slope:.4f} × 연도 {sign} {abs(intercept):.2f}")

# --- 100년에 몇 도 오르는가 ---
st.subheader("100년에 몇 도 오를까?")
st.markdown(
    f"""
    <div style="text-align:center; padding:0.8rem 0;">
        <div style="font-size:1.1rem; opacity:0.7;">전체 기간({start_year}~{end_year}년) 직선의 기울기</div>
        <div style="font-size:4rem; font-weight:800; line-height:1.2;">{slope * 100:+.2f} °C</div>
        <div style="font-size:1.1rem; opacity:0.7;">/ 100년</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- 전체 기간 vs 최근 N년 기울기 비교 ---
st.markdown(f"##### 전체 기간 vs 최근 {n_recent}년")
col_all, col_recent = st.columns(2)
with col_all:
    st.markdown(f"**전체 기간** ({start_year}~{end_year}년, {n_years}개 해)")
    st.metric("100년에", f"{slope * 100:+.2f} °C")
    st.caption(f"상관계수 r = {r:.3f} · 오차 ±{slope_se * 100:.2f} °C")
with col_recent:
    st.markdown(f"**최근 {n_recent}년** ({recent_start}~{end_year}년, {n_recent}개 해)")
    st.metric(
        "100년에",
        f"{slope_r * 100:+.2f} °C",
        delta=f"전체 기간보다 {(slope_r - slope) * 100:+.2f} °C",
        delta_color="off",
    )
    st.caption(f"상관계수 r = {r_recent:.3f} · 오차 ±{slope_r_se * 100:.2f} °C")
st.caption(
    "※ 오차는 기울기의 표준오차(100년 환산)입니다. 짧은 기간일수록 기울기가 크게 흔들리므로, "
    "최근 기울기는 '지금 속도'를 보여 줄 뿐 100년 뒤를 그대로 예측한 값은 아닙니다."
)

with st.expander("연도별 평균기온 표 보기"):
    st.dataframe(
        annual.rename(columns={"관측일수": "관측일수(일)", "연평균기온": "연평균기온(°C)"}),
        hide_index=True,
        width="stretch",
    )
