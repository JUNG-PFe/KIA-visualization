import pandas as pd
import streamlit as st
import json
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    data_url = "https://github.com/JUNG-PFe/pitcher-visualization_2/raw/refs/heads/main/combined_pitch_data.xlsx"
    df = pd.read_excel(data_url)
    return df

df = load_new_data()

st.set_page_config(
    page_title="24 호크아이 투수 피칭 궤적",
    page_icon="⚾",
    layout="wide"
)

# -------------------------------
# 로그인 여부 확인
# -------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("로그인 후에 이 페이지를 이용할 수 있습니다.")
    st.stop()

st.title("24 투수 피칭 궤적")

# 세션 상태 초기화
if "filter_applied" not in st.session_state:
    st.session_state.filter_applied = False

df['date'] = pd.to_datetime(df['date'], errors='coerce')
df = df.dropna(subset=['date'])  # 날짜 없는 데이터 제거
df['ball_pos_X'] = df['ball_pos_X'] * 100
df['ball_pos_Y'] = df['ball_pos_Y'] * 100
df['ball_pos_Z'] = df['ball_pos_Z'] * 100

# time 값 기준으로 그룹화 (time이 줄어드는 순간 새로운 그룹 생성)
df['time_diff'] = df['time'].diff()
df['group'] = (df['time_diff'] < 0).cumsum()

# Streamlit UI 구성
st.title("피칭궤적 시각화")
st.sidebar.header("Filter Options")

search_name = st.sidebar.text_input("Search Pitcher Name", "")
filtered_pitchers = df['pitcher'].unique()

if search_name:
    filtered_pitchers = [name for name in filtered_pitchers if search_name in name]

pitcher_selected = st.sidebar.selectbox("Select Pitcher", filtered_pitchers)
date_selected = st.sidebar.selectbox("Select Date", sorted(df[df['pitcher'] == pitcher_selected]['date'].unique()))
pitch_types_selected = st.sidebar.multiselect("Select Pitch Type(s)", df['pitch_type'].unique(), default=df['pitch_type'].unique())
zone_selected = st.sidebar.multiselect("Select Zone(s)", df['zone'].unique(), default=df['zone'].unique())

# 데이터 필터링
filtered_data = df[
    (df['pitcher'] == pitcher_selected) &
    (df['date'] == date_selected) &
    (df['pitch_type'].isin(pitch_types_selected)) &
    (df['zone'].isin(zone_selected))
]

# 시각화 생성
fig = go.Figure()

for group in filtered_data['group'].unique():
    group_data = filtered_data[filtered_data['group'] == group]
    for pitch_type in group_data['pitch_type'].unique():
        pitch_data = group_data[group_data['pitch_type'] == pitch_type]
        
        # 궤적 추가
        fig.add_trace(go.Scatter3d(
            x=pitch_data['ball_pos_X'],
            y=pitch_data['ball_pos_Y'],
            z=pitch_data['ball_pos_Z'],
            mode='lines',
            line=dict(width=4, color=cols[pitch_type]),  # 구종별 선 색상 적용
            name=f"{pitch_type} (Group {group})",
            legendgroup=f"{pitch_type} (Group {group})"  # 범례 그룹 지정
        ))

        # 마지막 점과 그 전 점 계산
        if len(pitch_data) > 1:
            last_point = pitch_data.iloc[-1]
            second_last_point = pitch_data.iloc[-2]

            # 기울기 계산
            delta_y = last_point['ball_pos_Y'] - second_last_point['ball_pos_Y']
            delta_x = last_point['ball_pos_X'] - second_last_point['ball_pos_X']
            delta_z = last_point['ball_pos_Z'] - second_last_point['ball_pos_Z']

            # y=150일 때 x와 z 값 계산
            if last_point['ball_pos_Y'] > 150:
                factor = (150 - last_point['ball_pos_Y']) / delta_y
                extended_x = last_point['ball_pos_X'] + delta_x * factor
                extended_z = last_point['ball_pos_Z'] + delta_z * factor

                # 익스텐션 선 추가 (legendgroup 동일하게 설정)
                fig.add_trace(go.Scatter3d(
                    x=[last_point['ball_pos_X'], extended_x],  # x 좌표
                    y=[last_point['ball_pos_Y'], 150],         # y 좌표
                    z=[last_point['ball_pos_Z'], extended_z],  # z 좌표
                    mode='lines',
                    line=dict(color=cols[pitch_type], width=4),  # 점선 스타일
                    name=f"{pitch_type} (Group {group})",  # 동일한 name
                    legendgroup=f"{pitch_type} (Group {group})",  # 동일한 legendgroup
                    showlegend=False  # 범례 표시 숨김
                ))

# 사각형 추가 (스트라이크 존)
x_range = [-23, 23]
z_range = [46, 105]
y_constant = 150  # 사각형은 홈 플레이트 근처(y=150)에서 추가

# 사각형 꼭짓점 좌표
rect_x = [x_range[0], x_range[1], x_range[1], x_range[0], x_range[0]]
rect_y = [y_constant] * 5
rect_z = [z_range[0], z_range[0], z_range[1], z_range[1], z_range[0]]

# 스트라이크 존 추가
fig.add_trace(go.Scatter3d(
    x=rect_x,
    y=rect_y,
    z=rect_z,
    mode='lines',
    line=dict(color='blue', width=4),
    name='Strike Zone'
))

# 투구판 추가
pitch_plate_x = [-30, 30, 30, -30, -30]  # x 좌표
pitch_plate_y = [1700] * 5               # y 좌표 (고정)
pitch_plate_z = [0, 0, 200, 200, 0]      # z 좌표

fig.add_trace(go.Scatter3d(
    x=pitch_plate_x,
    y=pitch_plate_y,
    z=pitch_plate_z,
    mode='lines',
    line=dict(color='darkgreen', width=5),
    name='Pitch Plate'
))

# 그래프 레이아웃 설정
fig.update_layout(
    scene=dict(
        xaxis=dict(title="X Position (Scaled)", range=[-200, 200]),
        yaxis=dict(title="Y Position (Scaled)", range=[100, 1700]),
        zaxis=dict(title="Z Position (Scaled)", range=[0, 200]),
        aspectratio=dict(x=2, y=4, z=1)
    ),
    title=f"{pitcher_selected} - {date_selected} - Selected Zones: {', '.join(map(str, zone_selected))}",
    margin=dict(l=0, r=0, b=0, t=40),
    showlegend=True,
    legend=dict(
        font=dict(size=14),  # 범례 글꼴 크기
        itemsizing="constant",  # 아이템 크기 일정하게 유지
        bgcolor="rgba(255,255,255,0.8)",  # 배경 투명도 조정
        bordercolor="black",  # 경계선 색
        borderwidth=1,  # 경계선 두께
    ),
    width=600,
    height=600
)

# Streamlit에 그래프 출력
st.plotly_chart(fig, use_container_width=True)

st.write("---")
st.write("범례에서 보고 싶은 것만 체크하여 확인 가능_한 구종당 범례")