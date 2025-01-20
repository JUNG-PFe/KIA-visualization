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

# 유효한 타격결과 정의
valid_hit_results = ["땅볼", "뜬공", "직선타", "병살타", "파울", "안타", "2루타", "3루타", "홈런", "희비"]
df = df[df['타격결과'].isin(valid_hit_results)]

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
    selected_hit_results = st.multiselect("타격결과 선택", ["전체"] + valid_hit_results, default=[])

# 구속 범위 필터
col9, col10 = st.columns(2)
with col9:
    min_speed = st.number_input("구속 최저 (km/h)", min_value=0, max_value=200, value=100)
with col10:
    max_speed = st.number_input("구속 최고 (km/h)", min_value=0, max_value=200, value=160)

def create_strike_zone_image(resize_width=300, resize_height=400):
    # 실제 스트라이크 존 경계선 값
    strike_zone_x_edges = np.linspace(-23, 23, 4)  # X축 경계선 (PlateLocSide)
    strike_zone_z_edges = np.linspace(46, 105, 4)  # Z축 경계선 (PlateLocHeight)

    # 이미지 크기 및 비율 설정
    width, height = 300, 400  # 이미지 기본 크기
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    # 폰트 설정
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    # 실제 경계선을 이미지 좌표로 변환 (정규화)
    x_edges = np.linspace(0, width, len(strike_zone_x_edges))
    z_edges = np.linspace(0, height, len(strike_zone_z_edges))

    # 스트라이크 존 그리기
    for z in range(3):
        for x in range(3):
            left = x_edges[x]
            right = x_edges[x + 1]
            top = z_edges[z]
            bottom = z_edges[z + 1]

            # 경계선 그리기
            draw.rectangle([left, top, right, bottom], outline="black", width=2)

            # 존 번호 표시
            zone_number = z * 3 + x + 1
            text_x = (left + right) / 2 - font_size // 2
            text_y = (top + bottom) / 2 - font_size // 2
            draw.text((text_x, text_y), str(zone_number), fill="black", font=font)

    # 이미지 크기 조정
    img = img.resize((resize_width, resize_height), Image.Resampling.LANCZOS)
    return img
col1, col2 = st.columns([1, 1])  # 동일한 비율로 두 열 생성

with col1:
    # 스트라이크 존 이미지 표시
    strike_zone_image = create_strike_zone_image(resize_width=300, resize_height=400)
    st.image(strike_zone_image, caption="스트라이크 존 (1~9)", use_container_width=False)

with col2:
    # 스트라이크 존 선택
    st.subheader("스트라이크 존 선택 (투수 시점)")
    cols = st.columns(3)  # 3개의 열로 선택 버튼 배치

    # 스트라이크 존 체크박스 생성
    selected_zones = []
    zones = [f"존 {i}" for i in range(1, 10)]
    for idx, zone in enumerate(zones):
        with cols[idx % 3]:  # 3개의 열로 나눠 배치
            if st.checkbox(zone, key=f"zone_{idx}"):
                selected_zones.append(zone)


# 검색 버튼
if st.button("검색 실행"):
    st.session_state.filter_applied = True

# 필터 적용 로직
if st.session_state.filter_applied:
    filtered_df = df.copy()

    # PlateLocSide 및 PlateLocHeight 값 변환
    filtered_df['PlateLocSide'] = pd.to_numeric(filtered_df['PlateLocSide'], errors='coerce') * 100
    filtered_df['PlateLocHeight'] = pd.to_numeric(filtered_df['PlateLocHeight'], errors='coerce') * 100
    filtered_df.dropna(subset=['PlateLocSide', 'PlateLocHeight'], inplace=True)

    # 날짜 범위 필터
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.Timestamp(start_date)) & 
                                  (filtered_df['Date'] <= pd.Timestamp(end_date))]

    # 연도 및 월 필터
    if selected_year != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.year == int(selected_year)]
    if selected_month != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.month == int(selected_month)]

    # 타자 이름 필터
    if batter_name and batter_name != "전체":
        filtered_df = filtered_df[filtered_df['타자'] == batter_name]

    # 투수 유형 필터
    if pitcher_type != "전체":
        filtered_df = filtered_df[filtered_df['투수유형'] == pitcher_type]

    # 주자 상황 필터
    if runner_status != "전체":
        if runner_status == "주자무":
            filtered_df = filtered_df[filtered_df['주자'] == "주자무"]
        else:
            filtered_df = filtered_df[filtered_df['주자'] != "주자무"]

    # 구종 필터
    if pitch_type:
        filtered_df = filtered_df[filtered_df['구종'].isin(pitch_type)]

    # 타격결과 필터
    if selected_hit_results and "전체" not in selected_hit_results:
        filtered_df = filtered_df[filtered_df['타격결과'].isin(selected_hit_results)]

    # 구속 필터
    filtered_df = filtered_df[(filtered_df['구속'] >= min_speed) & (filtered_df['구속'] <= max_speed)]

    # 스트라이크 존 필터
    if selected_zones and "전체" not in selected_zones:
        # 스트라이크 존 기준
        strike_zone_x_edges = np.linspace(-23, 23, 4)
        strike_zone_z_edges = np.linspace(46, 105, 4)

        # 선택된 존 번호를 필터링
        try:
            selected_zone_indices = [int(zone.split(" ")[1]) for zone in selected_zones]
        except (IndexError, ValueError):
            selected_zone_indices = []

        zone_conditions = []
        for zone_idx in selected_zone_indices:
            if 1 <= zone_idx <= 9:
                x_range = strike_zone_x_edges[(zone_idx - 1) % 3: (zone_idx - 1) % 3 + 2]
                z_range = strike_zone_z_edges[(zone_idx - 1) // 3: (zone_idx - 1) // 3 + 2]
                zone_conditions.append(
                    (filtered_df['PlateLocSide'] >= x_range[0]) &
                    (filtered_df['PlateLocSide'] <= x_range[1]) &
                    (filtered_df['PlateLocHeight'] >= z_range[0]) &
                    (filtered_df['PlateLocHeight'] <= z_range[1])
                )
        # 조건 병합 및 데이터 필터링
        if zone_conditions:
            filtered_df = filtered_df[np.logical_or.reduce(zone_conditions)]

    if not filtered_df.empty:
        st.subheader("기본 분석 값")

        filtered_df['타격결과'] = pd.Categorical(
            filtered_df['타격결과'],
            categories=valid_hit_results,
            ordered=True
        )

        analysis = filtered_df.groupby('타격결과').agg(
            타구수=('타격결과', 'count'),
            타구_비율=('타격결과', lambda x: round((x.count() / len(filtered_df)) * 100, 1)),
            평균_타구속도=('ExitSpeed', lambda x: round(x.mean(), 1)),
            평균_타구각=('Angle', lambda x: round(x.mean(), 1)),
            평균_타구방향=('Direction', lambda x: round(x.mean(), 1)),
            평균_회전수=('HitSpinRate', lambda x: round(x.mean(), 0)),
            평균_비거리=('LastTrackedDistance', lambda x: round(x.mean(), 1)),
        ).reset_index()
        analysis = analysis.sort_values(by='타격결과')

        st.dataframe(analysis)

        st.subheader("타구 Spray Chart")

    # 타격결과 필터링을 위한 선택
    selected_hit_types = st.multiselect(
        "표시할 타격 결과 선택",
        valid_hit_results,
        default=valid_hit_results  # 기본값으로 모든 결과 선택
    )

    # 선택된 타격결과만 필터링
    filtered_hits = filtered_df[filtered_df['타격결과'].isin(selected_hit_types)]

    # 타구 위치 시각화
    if not filtered_hits.empty:
        filtered_df = filtered_hits  # filtered_hits 데이터프레임 사용
        filtered_df['hit_X'] = filtered_df['hit_X'] * 1.0  # X좌표
        filtered_df['hit_Y'] = filtered_df['hit_Y'] * 1.0  # Y좌표

        def generate_outfield_curve():
            # 외야 끝점 좌표 설정
            left_corner = [-73, 73]
            center = [0, 125]
            right_corner = [73, 73]

            # Bézier 곡선을 생성하기 위한 파라미터 t
            t = np.linspace(0, 1, 100)  # 부드러운 곡선을 위해 100개의 점 생성

            control_left = [-50, 135]  # 왼쪽 보조 제어점
            control_right = [50, 135]  # 오른쪽 보조 제어점

            # Bézier 곡선의 x, y 좌표 계산 (4개의 점 사용)
            x_curve = (
                (1 - t)**3 * left_corner[0] +
                3 * (1 - t)**2 * t * control_left[0] +
                3 * (1 - t) * t**2 * control_right[0] +
                t**3 * right_corner[0]
            )
            y_curve = (
                (1 - t)**3 * left_corner[1] +
                3 * (1 - t)**2 * t * control_left[1] +
                3 * (1 - t) * t**2 * control_right[1] +
                t**3 * right_corner[1]
            )

            return x_curve, y_curve

        infield_diamond = [
            [19.098, 19.14],   # 1루
            [-0.024, 38.501],  # 2루
            [-19.124, 19.122], # 3루
            [0, 0],            # 홈
            [19.098, 19.14]    # 다시 1루로 닫음
        ]

        x_curve, y_curve = generate_outfield_curve()

        # 시각화 객체 생성
        fig = go.Figure()

        # 내야 다이아몬드 추가
        fig.add_trace(go.Scatter(
            x=[x[0] for x in infield_diamond],
            y=[x[1] for x in infield_diamond],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False  # 범례 숨김
        ))

        # 부드러운 외야 곡선 추가
        fig.add_trace(go.Scatter(
            x=x_curve,
            y=y_curve,
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False  # 범례 숨김
        ))

        # 곡선 양 끝점에서 홈으로 연결하는 직선 추가
        fig.add_trace(go.Scatter(
            x=[-73, 0],  # 왼쪽 끝점 -> 홈
            y=[73, 0],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False  # 범례 숨김
        ))
        fig.add_trace(go.Scatter(
            x=[73, 0],  # 오른쪽 끝점 -> 홈
            y=[73, 0],
            mode='lines',
            line=dict(color='black', width=2),
            showlegend=False  # 범례 숨김
        ))

        # 타구 데이터를 기준으로 색상 매핑
        hit_color_mapping = {
            "땅볼": "brown",
            "뜬공": "blue",
            "안타": "green",
            "2루타": "orange",
            "3루타": "purple",
            "홈런": "red"
        }

        for result in hit_color_mapping:
            subset = filtered_df[filtered_df['타격결과'] == result]
            fig.add_trace(go.Scatter(
                x=subset['hit_X'],
                y=subset['hit_Y'],
                mode='markers',
                marker=dict(color=hit_color_mapping[result], size=8, opacity=0.8),
                name=result,
                hovertemplate=(
                    "<b>타격결과:</b> %{text}<br>"
                    "<b>X:</b> %{x}<br>"
                    "<b>Y:</b> %{y}<br>"
                    "<b>Date:</b> %{customdata}<extra></extra>"
                ),
                text=subset['타격결과'],  # 타격결과를 텍스트로 추가
                customdata=subset['Date'].dt.strftime('%Y-%m-%d')  # Date를 포맷팅하여 추가
            ))
        # 그래프 레이아웃 설정
        fig.update_layout(
            title="Spray Chart",
            xaxis=dict(title="X 좌표", range=[-90, 90], zeroline=False),
            yaxis=dict(title="Y 좌표", range=[0, 130], zeroline=False),
            showlegend=True,
            width=800,
            height=800,  # 그래프 높이를 정사각형에 맞춰 설정
            plot_bgcolor="white",  # 배경색 설정
            margin=dict(l=50, r=50, t=50, b=50),  # 여백 설정
        )

        # X축과 Y축 비율 고정
        fig.update_yaxes(scaleanchor="x", scaleratio=1)

        # Streamlit에 그래프 표시
        st.plotly_chart(fig)

    else:
        st.warning("선택된 타격 결과에 해당하는 데이터가 없습니다.")

        # 타격 결과에 "파울" 추가
        hit_color_mapping = {
        "땅볼": "brown",
        "뜬공": "blue",
        "안타": "green",
        "2루타": "orange",
        "3루타": "purple",
        "홈런": "red",
        "파울": "gray"
    }

    

    # 플레이트 좌표 설정 (3D, 점 10개)
    plate_x = [0, 0, -21.59, -21.59, -21.59, -21.59, 21.59, 21.59, 21.59, 21.59]  # X축
    plate_y = [0, 0, 21.59, 21.59, 43.18, 43.18, 21.59, 21.59, 43.18, 43.18]      # Y축
    plate_z = [46, 104, 46, 104, 46, 104, 46, 104, 46, 104]                       # Z축

    # 밑면 점 순서 재정렬 (Z=46)
    bottom_x = [0, -21.59, -21.59, 21.59, 21.59, 0]
    bottom_y = [0, 21.59, 43.18, 43.18, 21.59, 0]
    bottom_z = [46, 46, 46, 46, 46, 46]

    # 윗면 점 순서 재정렬 (Z=104)
    top_x = [0, -21.59, -21.59, 21.59, 21.59, 0]
    top_y = [0, 21.59, 43.18, 43.18, 21.59, 0]
    top_z = [104, 104, 104, 104, 104, 104]

    if not filtered_hits.empty:
        selected_hit_types_xyz = st.multiselect(
            "3D 컨택 포인트 표시할 타격 결과 선택",
            list(hit_color_mapping.keys()),
            default=list(hit_color_mapping.keys()),  # 기본값으로 모든 결과 선택
            key="xyz_hit_selection"
        )

        # 선택된 타격결과만 필터링
        filtered_hits_xyz = filtered_hits[filtered_hits['타격결과'].isin(selected_hit_types_xyz)]

        if not filtered_hits_xyz.empty:
            fig_xyz = go.Figure()

            # 타격 데이터 추가
            for result in selected_hit_types_xyz:
                subset = filtered_hits_xyz[filtered_hits_xyz['타격결과'] == result]
                fig_xyz.add_trace(go.Scatter3d(
                    x=subset['ContactPositionZ'] * 100,  # ContactPositionZ: 가로 (X축)
                    y=subset['ContactPositionX'] * 100,  # ContactPositionX: 세로 (Y축)
                    z=subset['ContactPositionY'] * 100,  # ContactPositionY: 높이 (Z축)
                    mode='markers',
                    marker=dict(color=hit_color_mapping[result], size=4, opacity=0.8),
                    name=result,
                    hovertemplate=(
                        "<b>타격결과:</b> %{text}<br>"
                        "<b>Z(가로):</b> %{x}<br>"
                        "<b>X(세로):</b> %{y}<br>"
                        "<b>Y(높이):</b> %{z}<br>"
                        "<b>Date:</b> %{customdata}<extra></extra>"
                    ),
                    text=subset['타격결과'],
                    customdata=subset['Date'].dt.strftime('%Y-%m-%d')
                ))

            # 밑면 오각형 (Z=46)
            fig_xyz.add_trace(go.Scatter3d(
                x=bottom_x,
                y=bottom_y,
                z=bottom_z,
                mode='lines',
                line=dict(color='black', width=3),  # 검정색 선
                name="strike zone"
            ))

            # 윗면 오각형 (Z=104)
            fig_xyz.add_trace(go.Scatter3d(
                x=top_x,
                y=top_y,
                z=top_z,
                mode='lines',
                line=dict(color='black', width=3),  # 검정색 선
                showlegend=False
            ))

            # 밑면과 윗면을 연결하는 세로선
            for i in range(5):  # 닫힌 오각형이므로 5개의 세로선만 필요
                fig_xyz.add_trace(go.Scatter3d(
                    x=[bottom_x[i], top_x[i]],
                    y=[bottom_y[i], top_y[i]],
                    z=[bottom_z[i], top_z[i]],
                    mode='lines',
                    line=dict(color='black', width=3),  # 검정색 세로선
                    showlegend=False
                ))

            # 레이아웃 설정
            fig_xyz.update_layout(
                title="3D 플레이트와 타격 결과",
                scene=dict(
                    xaxis=dict(title="가로 (ContactPositionZ)", range=[-80, 80]),
                    yaxis=dict(title="세로 (ContactPositionX)", range=[-100, 150]),
                    zaxis=dict(title="높이 (ContactPositionY)", range=[0, 180]),
                    aspectmode='manual',  # 비율을 수동으로 설정
                    aspectratio=dict(x=2, y=2.8, z=2),  # 각 축 비율 조정 (x:y:z)
                ),
                showlegend=True,
                width=800,  # 크기 조정
                height=800,  # 크기 조정
                margin=dict(l=50, r=50, t=50, b=50),
            )

            # Streamlit에 그래프 표시
            st.plotly_chart(fig_xyz)
        else:
            st.warning("3D 시각화: 선택된 타격 결과에 해당하는 데이터가 없습니다.")

    col1, col2 = st.columns(2)

    plate_x = [-21.59, -21.59, 0, 21.59, 21.59]
    plate_z = [43.18, 21.59, 0, 21.59, 43.18]

    # X-Z 시각화 (플레이트 포함)
    with col1:
        if not filtered_hits.empty:
            selected_hit_types_xz = st.multiselect(
                "존 좌우 컨택 포인트 표시할 타격 결과 선택",
                list(hit_color_mapping.keys()),
                default=list(hit_color_mapping.keys()),  # 기본값으로 모든 결과 선택
                key="xz_hit_selection"  # 고유 키 추가
            )

            # 선택된 타격결과만 필터링
            filtered_hits_xz = filtered_hits[filtered_hits['타격결과'].isin(selected_hit_types_xz)]

            if not filtered_hits_xz.empty:
                fig_xz = go.Figure()

                # 타격 결과 데이터 추가 (값 변환 후 사용)
                for result in selected_hit_types_xz:
                    subset = filtered_hits_xz[filtered_hits_xz['타격결과'] == result]
                    fig_xz.add_trace(go.Scatter(
                        x=subset['ContactPositionZ'] * 100,  # X좌표 변환 (ContactPositionZ)
                        y=subset['ContactPositionX'] * 100,  # Y좌표 변환 (ContactPositionX)
                        mode='markers',
                        marker=dict(color=hit_color_mapping[result], size=8, opacity=0.8),
                        name=result,
                        hovertemplate=(
                            "<b>타격결과:</b> %{text}<br>"
                            "<b>Z:</b> %{x}<br>"
                            "<b>X:</b> %{y}<br>"
                            "<b>Date:</b> %{customdata}<extra></extra>"
                        ),
                        text=subset['타격결과'],  # '타격결과'를 툴팁에 추가
                        customdata=subset['Date'].dt.strftime('%Y-%m-%d')  # 날짜 데이터를 포맷팅하여 추가
                    ))

                # 플레이트 추가 (변환 없이 사용)
                fig_xz.add_trace(go.Scatter(
                    x=plate_x,
                    y=plate_z,
                    mode='lines',
                    line=dict(color='red', width=2),
                    showlegend=False  # 플레이트 범례 숨김
                ))

                # 그래프 레이아웃 설정
                fig_xz.update_layout(
                    title="존 좌우 컨택 포인트",
                    xaxis=dict(title="ContactPositionZ", range=[-50, 50], zeroline=False),  # * 100 변환 고려
                    yaxis=dict(title="ContactPositionX", range=[0, 100], zeroline=False),  # * 100 변환 고려
                    showlegend=True,
                    width=500,  # 크기 조정
                    height=500,  # 크기 조정
                    plot_bgcolor="white",
                    margin=dict(l=50, r=50, t=50, b=50)
                )

                # Streamlit에 그래프 표시
                st.plotly_chart(fig_xz)
            else:
                st.warning("X-Z 시각화: 선택된 타격 결과에 해당하는 데이터가 없습니다.")

    # X-Y 시각화 (플레이트 제외)
    with col2:
        if not filtered_hits.empty:
            selected_hit_types_xy = st.multiselect(
                "존 앞뒤 컨택 포인트 표시할 타격 결과 선택",
                list(hit_color_mapping.keys()),
                default=list(hit_color_mapping.keys()),  # 기본값으로 모든 결과 선택
                key="xy_hit_selection"  # 고유 키 추가
            )

            # 선택된 타격결과만 필터링
            filtered_hits_xy = filtered_hits[filtered_hits['타격결과'].isin(selected_hit_types_xy)]

            if not filtered_hits_xy.empty:
                fig_xy = go.Figure()

                # 선택된 타격 결과 데이터 추가 (값 변환 후 사용)
                for result in selected_hit_types_xy:
                    subset = filtered_hits_xy[filtered_hits_xy['타격결과'] == result]
                    fig_xy.add_trace(go.Scatter(
                        x=subset['ContactPositionX'] * 100,  # X좌표 변환
                        y=subset['ContactPositionY'] * 100,  # Y좌표 변환
                        mode='markers',
                        marker=dict(color=hit_color_mapping[result], size=8, opacity=0.8),
                        name=result,
                        hovertemplate=(
                            "<b>타격결과:</b> %{text}<br>"  # 결과 표시
                            "<b>X:</b> %{x}<br>"  # X값
                            "<b>Y:</b> %{y}<br>"  # Y값
                            "<b>Date:</b> %{customdata}<extra></extra>"  # 날짜
                        ),
                        text=subset['타격결과'],  # '타격결과'를 표시
                        customdata=subset['Date'].dt.strftime('%Y-%m-%d')  # 'Date'를 포맷팅하여 표시
                    ))


                # 플레이트 상자 추가
                plate_box_x = [0, 0, 43.18, 43.18, 0]  # X 좌표
                plate_box_y = [46, 104, 104, 46, 46]  # Y 좌표
                fig_xy.add_trace(go.Scatter(
                    x=plate_box_x,
                    y=plate_box_y,
                    mode='lines',
                    line=dict(color='red', width=2),
                    name='Plate Box',
                    showlegend=False
                ))
                dot_line_x = [21.59, 21.59]  # 점선 X 좌표
                dot_line_y = [46, 104]       # 점선 Y 좌표
                fig_xy.add_trace(go.Scatter(
                    x=dot_line_x,
                    y=dot_line_y,
                    mode='lines',
                    line=dict(color='red', width=1, dash='dot'),  # 점선 설정
                    name='Center Line',
                    showlegend=False
                ))

                # 그래프 레이아웃 설정
                fig_xy.update_layout(
                    title="존 앞뒤 컨택 포인트",
                    xaxis=dict(title="ContactPositionX", range=[-20, 120], zeroline=False),
                    yaxis=dict(title="ContactPositionY", range=[20, 120], zeroline=False),
                    showlegend=True,
                    width=700,  # 크기 조정
                    height=500,  # 크기 조정
                    plot_bgcolor="white",
                    margin=dict(l=50, r=50, t=50, b=50)
                )

                # Streamlit에 그래프 표시
                st.plotly_chart(fig_xy)
            else:
                st.warning("X-Y 시각화: 선택된 타격 결과에 해당하는 데이터가 없습니다.")

if not filtered_df.empty:
    filtered_df_1 = filtered_df[filtered_df['타격결과'].isin(valid_hit_results)].copy()

    
    # 선택된 지표에 따라 필터링된 데이터 사용
    selected_result = st.selectbox(
        "타격 결과 선택",
        ["전체"] + valid_hit_results,
        index=0
    )

    # 선택된 결과에 따라 필터링
    if selected_result != "전체":
        filtered_df_2d = filtered_df_1[filtered_df_1['타격결과'] == selected_result]
    else:
        filtered_df_2d = filtered_df_1

    # 데이터가 비어 있는 경우 경고 메시지 출력
    if filtered_df_2d.empty:
        st.warning("선택된 타격 결과에 해당하는 데이터가 없습니다.")
    else:
        # 산점도 생성
        fig_2d = px.scatter(
            filtered_df_2d,
            x='ExitSpeed',  # X축
            y='Angle',      # Y축
            color='타격결과',  # 색상으로 타격 결과 구분
            category_orders={"타격결과": valid_hit_results},  # Plotly에서 명시적 정렬
            title=f"2D 산점도: 타구 속도와 각도 ({selected_result})",
            labels={
                'ExitSpeed': '타구 속도',
                'Angle': '타구 각도'
            },
            opacity=0.8,  # 투명도 설정
            hover_data=['Date'] 
        )

        fig_2d.update_traces(marker=dict(size=10)) 

        # 레이아웃 설정
        fig_2d.update_layout(
            xaxis=dict(
                title='타구 속도',
                title_font=dict(size=14, color='black'),
                showline=True,
                linewidth=2,
                linecolor='black',
                range=[20, 190]
            ),
            yaxis=dict(
                title='타구 각도',
                title_font=dict(size=14, color='black'),
                showline=True,
                linewidth=2,
                linecolor='black',
                range=[-50, 90]
            )
        )

        for y_value in [0, 10, 20, 28]:
            fig_2d.add_shape(
                type="line",
                x0=fig_2d.layout.xaxis.range[0] if fig_2d.layout.xaxis.range else filtered_df_2d['PTS_ExitSpeed'].min(),  # X축 시작
                x1=fig_2d.layout.xaxis.range[1] if fig_2d.layout.xaxis.range else filtered_df_2d['PTS_ExitSpeed'].max(),  # X축 끝
                y0=y_value,  # Y축 값
                y1=y_value,  # 동일한 Y축 값
                line=dict(color="black", width=2)  # 선 색상과 두께
            )
        for x_value in [140, 160]:
            fig_2d.add_shape(
                type="line",
                x0=x_value,  # X축 값
                x1=x_value,  # 동일한 X축 값
                y0=fig_2d.layout.yaxis.range[0] if fig_2d.layout.yaxis.range else filtered_df_2d['PTS_Angle'].min(),  # Y축 시작
                y1=fig_2d.layout.yaxis.range[1] if fig_2d.layout.yaxis.range else filtered_df_2d['PTS_Angle'].max(),  # Y축 끝
                line=dict(color="black", width=2)  # 선 색상과 두께
            )

        
        fig_2d.update_layout(
            width=200,  # 원하는 폭 (px 단위)
            height=700  # 원하는 높이 (px 단위)
        )

        # Streamlit에서 그래프 표시
        st.plotly_chart(fig_2d, use_container_width=True)
            


    if not filtered_df.empty:
        # 필터링된 데이터 표시
        st.write("### 필터링된 데이터")
        st.write(f"총 데이터 수: **{len(filtered_df)}** 행")
        st.dataframe(filtered_df)

        # 데이터 다운로드
        st.subheader("결과 다운로드")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            filtered_df.to_excel(writer, index=False, sheet_name='Filtered Data')
            # writer.save()는 필요 없음
        output.seek(0)  # 스트림의 시작 위치로 이동

        st.download_button(
            label="필터링된 데이터 다운로드 (Excel)",
            data=output,
            file_name='filtered_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
    else:
        st.warning("필터링된 데이터가 없습니다.")