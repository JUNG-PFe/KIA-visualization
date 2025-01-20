import pandas as pd
import streamlit as st
import plotly.express as px
import numpy as np
import io

# 데이터 컬러 설정
cols = {
    "직구": "#4C569B",
    "투심": "#B590C3",
    "커터": "#45B0D8",
    "슬라": "firebrick",
    "스위퍼": "#00FF00",
    "체인": "#FBE25E",
    "포크": "MediumSeaGreen",
    "커브": "orange",
    "너클": "black"
}

@st.cache_data
def load_new_data():
    data_url = "https://github.com/JUNG-PFe/pitcher-visualization_2/raw/refs/heads/main/PTS%202024%20%EC%A0%84%EA%B2%BD%EA%B8%B0_%EC%88%98%EC%A0%95.xlsx"
    df = pd.read_excel(data_url)
    return df

df = load_new_data()

# 날짜 처리
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# 페이지 설정 (스크립트의 맨 위에 위치해야 함)
st.set_page_config(
    page_title="24 PTS 투수 존별 타구속도",
    page_icon="⚾",
    layout="wide"
)

# -------------------------------
# 로그인 여부 확인
# -------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("로그인 후에 이 페이지를 이용할 수 있습니다.")
    st.stop()  # 로그인 상태가 아니면 여기서 실행 중지

st.title("24 PTS 투수 존별 타구속도")

# 세션 상태 초기화
if "filter_applied" not in st.session_state:
    st.session_state.filter_applied = False

# 데이터 필터링 섹션
st.subheader("데이터 필터링")

# -------------------
# 연도 및 월 필터 가로 배치
# -------------------
st.subheader("연도 및 월 필터")
col1, col2 = st.columns(2)
with col1:
    unique_years = sorted(df['Date'].dt.year.unique())
    selected_year = st.selectbox("연도 선택", ["전체"] + unique_years)
with col2:
    unique_months = ["전체"] + list(range(1, 13))
    selected_month = st.selectbox("월 선택", unique_months)

# -------------------
# 날짜 범위 필터 (길게 표시)
# -------------------
st.subheader("날짜 범위 필터")
date_range = st.date_input(
    "날짜 범위",
    [df['Date'].min(), df['Date'].max()],  # 기본값 설정
    key="date_range",
    label_visibility="visible",
    help="필터링에 사용할 시작 날짜와 종료 날짜를 선택하세요."
)

# -------------------
# 투수 이름 검색 및 선택 가로 배치
# -------------------
st.subheader("투수 이름 필터")
col3, col4 = st.columns(2)
with col3:
    pitcher_search_query = st.text_input("투수 이름 검색", "").strip()
    if pitcher_search_query:
        pitcher_suggestions = [
            name for name in sorted(df['Pitcher'].unique()) 
            if pitcher_search_query.lower() in name.lower()
        ]
    else:
        pitcher_suggestions = sorted(df['Pitcher'].unique())
with col4:
    if pitcher_suggestions:
        pitcher_name = st.selectbox("투수 이름 선택", pitcher_suggestions)
    else:
        pitcher_name = None

# -------------------
# 구종 필터
# -------------------
st.subheader("구종 필터")
pitch_type = st.multiselect(
    "구종 선택",
    df['PitchType'].unique(),
    help="원하는 구종을 선택하세요."
)

# 검색 버튼
if st.button("검색 실행"):
    st.session_state.filter_applied = True

# 필터 적용 로직
if st.session_state.filter_applied:
    filtered_df = df.copy()

    # 날짜 필터
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.Timestamp(start_date)) & (filtered_df['Date'] <= pd.Timestamp(end_date))]

    if selected_year != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.year == selected_year]
    if selected_month != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.month == selected_month]

    # 투수 이름 필터
    if pitcher_name:
        filtered_df = filtered_df[filtered_df['Pitcher'] == pitcher_name]

    # 구종 필터
    if pitch_type:
        filtered_df = filtered_df[filtered_df['PitchType'].isin(pitch_type)]

    # PTS 데이터 전처리
    filtered_df['PTS_location_X'] = pd.to_numeric(filtered_df['PTS_location_X'], errors='coerce')
    filtered_df['PTS_location_Z'] = pd.to_numeric(filtered_df['PTS_location_Z'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['PTS_location_X', 'PTS_location_Z'])

    if not filtered_df.empty:
        # H 데이터 필터링
        filtered_h = filtered_df[filtered_df['PitchCall'] == 'H']
        filtered_h = filtered_h.dropna(subset=['PTS_location_X', 'PTS_location_Z', 'PTS_ExitSpeed'])

        if filtered_h.empty:
            st.warning("H 데이터가 없습니다.")
        else:
            st.subheader("타구 데이터 분석 및 시각화")

            # 스트라이크 존 9등분
            strike_zone_x_edges = np.linspace(-23, 23, 4)  # X축 경계선
            strike_zone_z_edges = np.linspace(46, 105, 4)  # Z축 경계선

            # 9등분 영역 계산
            filtered_h['x_bin'] = pd.cut(filtered_h['PTS_location_X'], bins=strike_zone_x_edges, labels=[1, 2, 3])
            filtered_h['z_bin'] = pd.cut(filtered_h['PTS_location_Z'], bins=strike_zone_z_edges, labels=[1, 2, 3])

            # 평균 ExitSpeed 계산
            heatmap_data = (
                filtered_h.groupby(['x_bin', 'z_bin'])
                .agg({'PTS_ExitSpeed': 'mean'})
                .reset_index()
            )
            heatmap_data['PTS_ExitSpeed'] = heatmap_data['PTS_ExitSpeed'].round(1)  # 소수점 한 자리로 반올림

            # 9등분 시각화
            heatmap_pivot = heatmap_data.pivot(index='z_bin', columns='x_bin', values='PTS_ExitSpeed')
            fig_heatmap = px.imshow(
                heatmap_pivot,
                text_auto=True,
                labels=dict(x="X 구역 (좌우)", y="Z 구역 (상하)", color="Exit Speed (km/h)"),
                title="스트라이크 존 9등분 Exit Speed 평균",
                color_continuous_scale="RdYlBu",
            )
            fig_heatmap.update_layout(
                xaxis=dict(side="top"),  # X축 레이블을 위쪽으로 배치
                width=700,
                height=700,
            )

            st.plotly_chart(fig_heatmap)

            # 스트라이크 존 내 각 영역 데이터 다운로드
            output_heatmap = io.BytesIO()
            with pd.ExcelWriter(output_heatmap, engine='xlsxwriter') as writer:
                heatmap_data.to_excel(writer, index=False, sheet_name='Heatmap Data')
            output_heatmap.seek(0)

            st.download_button(
                label="H 데이터 Heatmap 다운로드 (Excel)",
                data=output_heatmap,
                file_name='heatmap_data.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )



