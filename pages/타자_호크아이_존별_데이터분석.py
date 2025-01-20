import pandas as pd
import streamlit as st
import numpy as np
import plotly.express as px
import io
from PIL import Image, ImageDraw, ImageFont
import plotly.graph_objects as go
import re

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
def load_data():
    # 데이터 URL
    data_url1 = "https://github.com/JUNG-PFe/Batter_visualization/raw/refs/heads/main/23_merged_data_%EC%88%98%EC%A0%95.xlsx"
    data_url2 = "https://github.com/JUNG-PFe/Batter_visualization/raw/refs/heads/main/24_merged_data_%EC%88%98%EC%A0%95.xlsx"
    
    # 데이터 로드
    df1 = pd.read_excel(data_url1)
    df2 = pd.read_excel(data_url2)
    
    # 날짜 형식 통일
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    # 병합
    combined_df = pd.concat([df1, df2], ignore_index=True)

    # 타격결과 매핑
    result_mapping = {
        "BB": "사구",
        "FL": "뜬공",
        "GR": "땅볼",
        "H1": "안타",
        "H2": "2루타",
        "H3": "3루타",
        "HR": "홈런",
        "DP": "병살타",
        "FO": "파울",
        "HI": "사구",
        "KK": "삼진",
        "SF": "희비",
        "LD": "직선타",
    }
    combined_df['타격결과'] = combined_df['타격결과'].map(result_mapping).fillna(combined_df['타격결과'])

    return combined_df

# 데이터 로드
df = load_data()

st.set_page_config(
    page_title="23-24 호크아이 타자 데이터 필터링 및 분석 앱",
    page_icon="⚾",
    layout="wide"
)

# -------------------------------
# 로그인 여부 확인
# -------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("로그인 후에 이 페이지를 이용할 수 있습니다.")
    st.stop()

st.title("23-24 호크아이 타자 데이터 분석")

# 세션 상태 초기화
if "filter_applied" not in st.session_state:
    st.session_state.filter_applied = False

# -------------------
# 필터링 설정
# -------------------
st.subheader("데이터 필터링")

# 연도 및 월 필터
col1, col2 = st.columns(2)
with col1:
    unique_years = sorted(df['Date'].dt.year.unique())
    selected_year = st.selectbox("연도 선택", ["전체"] + unique_years)
with col2:
    unique_months = ["전체"] + list(range(1, 13))
    selected_month = st.selectbox("월 선택", unique_months)

# 날짜 범위 필터
date_range = st.date_input(
    "날짜 범위",
    [df['Date'].min(), df['Date'].max()],
    key="date_range"
)

# 타자 이름 필터
col3, col4 = st.columns(2)

with col3:
    search_query = st.text_input("타자 이름 검색", "").strip()
    if search_query:
        suggestions = [name for name in sorted(df['타자'].unique()) if search_query.lower() in name.lower()]
    else:
        suggestions = sorted(df['타자'].unique())

with col4:
    # 기본값으로 2번째 항목을 선택 (index=1: '전체'는 index 0, suggestions의 두 번째 항목은 index 1)
    if len(suggestions) >= 2:
        batter_name = st.selectbox("타자 이름 선택", ["전체"] + suggestions, index=2)
    else:
        batter_name = st.selectbox("타자 이름 선택", ["전체"] + suggestions)


# 투수 유형 및 주자 상황 필터
col5, col6 = st.columns(2)
with col5:
    pitcher_type = st.selectbox("투수 유형 선택", ["전체", "우투", "좌투"])
with col6:
    runner_status = st.selectbox("주자 상황 선택", ["전체", "주자무", "나머지"])

# 구종 및 타격결과 필터
col7, col8 = st.columns(2)
with col7:
    pitch_type = st.multiselect("구종 선택", df['구종'].unique())
with col8:
    # 타격결과 필터링 추가
    available_hit_results = sorted(df['타격결과'].dropna().unique())
    selected_hit_results = st.multiselect("타격결과 선택", ["전체"] + available_hit_results, default=[])

# 구속 범위 필터
col9, col10 = st.columns(2)
with col9:
    min_speed = st.number_input("구속 최저 (km/h)", min_value=0, max_value=200, value=100)
with col10:
    max_speed = st.number_input("구속 최고 (km/h)", min_value=0, max_value=200, value=150)

# 검색 버튼
if st.button("검색 실행"):
    st.session_state.filter_applied = True

# -------------------
# 필터링 로직
# -------------------
def apply_filters(df, filters):
    filtered_df = df.copy()

    # 연도 필터
    if filters['selected_year'] != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.year == filters['selected_year']]

    # 월 필터
    if filters['selected_month'] != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.month == filters['selected_month']]

    # 날짜 범위 필터
    filtered_df = filtered_df[
        (filtered_df['Date'] >= pd.to_datetime(filters['date_range'][0])) &
        (filtered_df['Date'] <= pd.to_datetime(filters['date_range'][1]))
    ]

    # 타자 이름 필터
    if filters['batter_name'] != "전체":
        filtered_df = filtered_df[filtered_df['타자'] == filters['batter_name']]

    # 투수 유형 필터
    if filters['pitcher_type'] != "전체":
        filtered_df = filtered_df[filtered_df['투수유형'] == filters['pitcher_type']]

    # 주자 상황 필터
    if filters['runner_status'] != "전체":
        if filters['runner_status'] == "주자무":
            filtered_df = filtered_df[filtered_df['주자상황'] == "주자무"]
        else:
            filtered_df = filtered_df[filtered_df['주자상황'] != "주자무"]

    # 구종 필터
    if filters['pitch_type']:
        filtered_df = filtered_df[filtered_df['구종'].isin(filters['pitch_type'])]

    # 타격결과 필터
    if "전체" not in filters['selected_hit_results'] and filters['selected_hit_results']:
        filtered_df = filtered_df[filtered_df['타격결과'].isin(filters['selected_hit_results'])]

    # 구속 범위 필터
    filtered_df = filtered_df[
        (filtered_df['구속'] >= filters['min_speed']) & (filtered_df['구속'] <= filters['max_speed'])
    ]

    return filtered_df


# 데이터 준비 및 필터 적용
if st.session_state.filter_applied:
    filters = {
        'selected_year': selected_year,
        'selected_month': selected_month,
        'date_range': date_range,
        'batter_name': batter_name,
        'pitcher_type': pitcher_type,
        'runner_status': runner_status,
        'pitch_type': pitch_type,
        'selected_hit_results': selected_hit_results,
        'min_speed': min_speed,
        'max_speed': max_speed
    }

    filtered_df = apply_filters(df, filters)

    # 스트라이크 존 좌표 변환 및 Zone 생성
    filtered_df['PlateLocSide'] = pd.to_numeric(filtered_df['PlateLocSide'], errors='coerce') * 100
    filtered_df['PlateLocHeight'] = pd.to_numeric(filtered_df['PlateLocHeight'], errors='coerce') * 100
    filtered_df.dropna(subset=['PlateLocSide', 'PlateLocHeight'], inplace=True)

    # 히트와 장타 정의
    filtered_df['히트'] = filtered_df['타격결과'].isin(['안타', '2루타', '3루타', '홈런']).astype(int)
    filtered_df['장타'] = filtered_df['타격결과'].isin(['2루타', '3루타', '홈런']).astype(int)

    # 첫 번째 스트라이크 존 시각화
    strike_zone_x_edges = np.linspace(-23, 23, 4)
    strike_zone_z_edges = np.linspace(46, 105, 4)

    filtered_df['Zone'] = pd.cut(
        filtered_df['PlateLocSide'], bins=strike_zone_x_edges, labels=[1, 2, 3]
    ).astype(str) + pd.cut(
        filtered_df['PlateLocHeight'], bins=strike_zone_z_edges, labels=[1, 2, 3]
    ).astype(str)

    metric = st.selectbox("첫 번째 분석 지표 선택", ['히트', '장타', '인플레이 타구', '파울', '스윙중 헛스윙', '뜬공', '땅볼'], key='metric_1')

    metric_mapping = {
        '인플레이 타구': 'H',
        '파울': 'F',
        '스윙중 헛스윙': 'S',
        '히트': '히트',
        '장타': '장타'
    }
    metric = metric_mapping.get(metric, metric)

    if metric in ['H', 'F', 'S']:
        zone_summary = (
            filtered_df.groupby('Zone')['심판콜']
            .value_counts(normalize=False)
            .unstack(fill_value=0)
            .reindex(columns=['F', 'H', 'S'], fill_value=0)
        )
        total_counts = zone_summary.sum(axis=1)
        selected_counts = zone_summary[metric]
        selected_rate = selected_counts / total_counts
    elif metric in ['히트', '장타']:
        total_counts = filtered_df.groupby('Zone').size()
        selected_counts = filtered_df.groupby('Zone')[metric].sum()
        selected_rate = selected_counts / total_counts
    else:
        total_counts = filtered_df.groupby('Zone').size()
        selected_counts = (
            filtered_df.groupby('Zone')['타격결과']
            .apply(lambda x: (x == metric).sum())
        )
        selected_rate = selected_counts / total_counts

    all_zones = [f"{x}{y}" for x in [1, 2, 3] for y in [1, 2, 3]]
    selected_rate = selected_rate.reindex(all_zones, fill_value=0)
    selected_counts = selected_counts.reindex(all_zones, fill_value=0)
    total_counts = total_counts.reindex(all_zones, fill_value=0)

    zone_matrix = selected_rate.values.reshape(3, 3)
    total_matrix = total_counts.values.reshape(3, 3)
    selected_matrix = selected_counts.values.reshape(3, 3)

    min_val = zone_matrix.min()
    max_val = zone_matrix.max()
    threshold = min_val + (max_val - min_val) * 0.6

    # Y축에 맞게 데이터를 뒤집기
    zone_matrix = zone_matrix[::-1]
    total_matrix = total_matrix[::-1]
    selected_matrix = selected_matrix[::-1]

    fig1 = px.imshow(
        zone_matrix,
        labels=dict(x="좌/우", y="높음/낮음", color=f"{metric} 비율 (%)"),
        x=['좌', '중', '우'],  # X축 이름 유지
        y=['높음', '중간', '낮음'],  # Y축 순서를 변경
        color_continuous_scale="Reds",
        zmin=zone_matrix.min(),
        zmax=zone_matrix.max()
    )

    # Y축 순서 명시
    fig1.update_yaxes(
        categoryarray=['높음', '중간', '낮음'],  # Y축 순서 명시
        categoryorder='array'
    )

    # 값 추가
    for i in range(zone_matrix.shape[0]):
        for j in range(zone_matrix.shape[1]):
            percentage = (zone_matrix[i, j] * 100)
            success = int(selected_matrix[i, j])
            total = int(total_matrix[i, j])
            text_color = "white" if zone_matrix[i, j] > threshold else "black"

            fig1.add_annotation(
                text=f"{percentage:.1f}%<br>({success}/{total})",
                x=j, y=i, showarrow=False,
                font=dict(color=text_color, size=12)
            )

    st.plotly_chart(fig1, use_container_width=True)

    # 두 번째 스트라이크 존 시각화
    strike_zone_x_edges = [-28, -23, -18, 18, 23, 28]
    strike_zone_z_edges = [36, 46, 56, 95, 105, 115]

    filtered_df['Zone'] = pd.cut(
        filtered_df['PlateLocSide'], bins=strike_zone_x_edges, labels=[1, 2, 3, 4, 5]
    ).astype(str) + pd.cut(
        filtered_df['PlateLocHeight'], bins=strike_zone_z_edges, labels=[1, 2, 3, 4, 5]
    ).astype(str)

    metric = st.selectbox("두 번째 분석 지표 선택", ['히트', '장타', '인플레이 타구', '파울', '스윙중 헛스윙', '뜬공', '땅볼'], key='metric_2')

    metric = metric_mapping.get(metric, metric)

    if metric in ['H', 'F', 'S']:
        zone_summary = (
            filtered_df.groupby('Zone')['심판콜']
            .value_counts(normalize=False)
            .unstack(fill_value=0)
            .reindex(columns=['F', 'H', 'S'], fill_value=0)
        )
        total_counts = zone_summary.sum(axis=1)
        selected_counts = zone_summary[metric]
        selected_rate = selected_counts / total_counts
    elif metric in ['히트', '장타']:
        total_counts = filtered_df.groupby('Zone').size()
        selected_counts = filtered_df.groupby('Zone')[metric].sum()
        selected_rate = selected_counts / total_counts
    else:
        total_counts = filtered_df.groupby('Zone').size()
        selected_counts = (
            filtered_df.groupby('Zone')['타격결과']
            .apply(lambda x: (x == metric).sum())
        )
        selected_rate = selected_counts / total_counts

    all_zones = [f"{x}{y}" for x in [1, 2, 3, 4, 5] for y in [1, 2, 3, 4, 5]]
    selected_rate = selected_rate.reindex(all_zones, fill_value=0)
    selected_counts = selected_counts.reindex(all_zones, fill_value=0)
    total_counts = total_counts.reindex(all_zones, fill_value=0)

    zone_matrix = selected_rate.values.reshape(5, 5)
    total_matrix = total_counts.values.reshape(5, 5)
    selected_matrix = selected_counts.values.reshape(5, 5)

    min_val =  np.nanmin(zone_matrix)
    max_val = np.nanmax(zone_matrix)
    threshold = min_val + (max_val - min_val) * 0.6

    # Y축에 맞게 데이터를 뒤집기
    zone_matrix = zone_matrix[::-1]
    total_matrix = total_matrix[::-1]
    selected_matrix = selected_matrix[::-1]

    fig2 = px.imshow(
        zone_matrix,
        labels=dict(x="좌/우", y="높음/낮음", color=f"{metric} 비율 (%)"),
        x=['좌-외곽', '좌-중간', '중앙', '우-중간', '우-외곽'],  # X축 이름 유지
        y=['위-외곽', '위-중간', '중간', '아래-중간', '아래-외곽'],  # Y축 순서를 변경
        color_continuous_scale="Reds",
        zmin=zone_matrix.min(),
        zmax=zone_matrix.max()
    )

    # Y축 순서 명시
    fig2.update_yaxes(
        categoryarray=['위-외곽', '위-중간', '중간', '아래-중간', '아래-외곽'],  # Y축 순서 명시
        categoryorder='array'
    )

    # 값 추가
    for i in range(zone_matrix.shape[0]):
        for j in range(zone_matrix.shape[1]):
            percentage = (zone_matrix[i, j] * 100)
            success = int(selected_matrix[i, j])
            total = int(total_matrix[i, j])
            text_color = "white" if zone_matrix[i, j] > threshold else "black"

            fig2.add_annotation(
                text=f"{percentage:.1f}%<br>({success}/{total})",
                x=j, y=i, showarrow=False,
                font=dict(color=text_color, size=12)
            )

    fig2.update_layout(
        width=800,  # 원하는 폭 (px 단위)
        height=800  # 원하는 높이 (px 단위)
    )

    fig2.add_shape(
        type="rect",
        x0=0.5, x1=3.5, y0=0.5, y1=3.5,
        line=dict(color="black", width=3)
    )

    fig2.update_layout(
        width=800,  # 원하는 폭 (px 단위)
        height=800  # 원하는 높이 (px 단위)
    )

    fig2.add_shape(
        type="rect",
        x0=0.5, x1=3.5, y0=0.5, y1=3.5,
        line=dict(color="black", width=3)
    )

    st.plotly_chart(fig2, use_container_width=True)