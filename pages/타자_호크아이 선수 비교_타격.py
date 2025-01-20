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
    page_title="23-24 호크아이 데이터 선수간 비교",
    page_icon="⚾",
    layout="wide"
)

# -------------------------------
# 로그인 여부 확인
# -------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("로그인 후에 이 페이지를 이용할 수 있습니다.")
    st.stop()  # 로그인 상태가 아니면 여기서 실행 중지

st.title("호크아이 데이터 선수 간 비교분석")

# 탭 생성
tab1, tab2 = st.tabs(["선수 간 비교", "기간 간 비교"])

# -------------------
# Tab 1: 선수 간 비교
# -------------------
with tab1:
    st.subheader("선수 간 비교")

    # 선수 1과 선수 2 검색 및 선택 가로 배치
    col1, col2 = st.columns(2)

    with col1:
        search_query_1 = st.text_input("선수 1 검색", key="search_query_1").strip()
        if search_query_1:
            suggestions_1 = [name for name in sorted(df['타자'].unique()) if search_query_1.lower() in name.lower()]
        else:
            suggestions_1 = sorted(df['타자'].unique())

        # suggestions_1의 두 번째 항목을 기본 선택하도록 설정
        if len(suggestions_1) >= 2:
            pitcher1 = st.selectbox("선수 1 선택", suggestions_1, index=1, key="pitcher1")
        elif len(suggestions_1) == 1:
            pitcher1 = st.selectbox("선수 1 선택", suggestions_1, index=0, key="pitcher1")
        else:
            pitcher1 = None

    with col2:
        search_query_2 = st.text_input("선수 2 검색", key="search_query_2").strip()
        if search_query_2:
            suggestions_2 = [name for name in sorted(df['타자'].unique()) if search_query_2.lower() in name.lower()]
        else:
            suggestions_2 = sorted(df['타자'].unique())

        # suggestions_2의 두 번째 항목을 기본 선택하도록 설정
        if len(suggestions_2) >= 2:
            pitcher2 = st.selectbox("선수 2 선택", suggestions_2, index=1, key="pitcher2")
        elif len(suggestions_2) == 1:
            pitcher2 = st.selectbox("선수 2 선택", suggestions_2, index=0, key="pitcher2")
        else:
            pitcher2 = None

    # 구종 선택 및 타격 결과 선택
    col1, col2 = st.columns(2)

    with col1:
        pitch_type = st.multiselect(
            "구종 선택", 
            options=df['구종'].unique(), 
            default=list(df['구종'].unique()),  # 기본값: 전체 선택
            key="pitch_type"
        )

    with col2:
        hit_color_mapping = {
            "땅볼": "brown",
            "뜬공": "blue",
            "안타": "green",
            "2루타": "orange",
            "3루타": "purple",
            "홈런": "red",
            "파울": "gray"
        }
        valid_hit_results = list(hit_color_mapping.keys())
        selected_hit_results = st.multiselect(
            "타격 결과 선택", 
            valid_hit_results, 
            default=valid_hit_results,  # 기본값: 전체 선택
            key="hit_results"
        )

    if pitcher1 and pitcher2 and pitch_type and selected_hit_results:
        filtered_hits_1 = df[
            (df['타자'] == pitcher1) &
            (df['구종'].isin(pitch_type)) &
            (df['타격결과'].isin(selected_hit_results))
        ]
        filtered_hits_2 = df[
            (df['타자'] == pitcher2) &
            (df['구종'].isin(pitch_type)) &
            (df['타격결과'].isin(selected_hit_results))
        ]

        def plot_spray_chart(filtered_hits, player_name):
            # 빈 데이터프레임 확인
            if filtered_hits.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"{player_name}의 데이터가 없습니다.",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16),
                    xref="paper", yref="paper"
                )
                fig.update_layout(
                    title=f"{player_name} Spray Chart",
                    xaxis=dict(title="X 좌표", range=[-90, 90], zeroline=False),
                    yaxis=dict(title="Y 좌표", range=[0, 130], zeroline=False),
                    width=600, height=600,
                    plot_bgcolor="white",
                    margin=dict(l=50, r=50, t=50, b=50),
                )
                return fig

            # 타구 위치 시각화 데이터 준비
            filtered_hits['hit_X'] = filtered_hits['hit_X'] * 1.0
            filtered_hits['hit_Y'] = filtered_hits['hit_Y'] * 1.0

            # 외야 곡선 생성
            def generate_outfield_curve():
                left_corner = [-73, 73]
                center = [0, 125]
                right_corner = [73, 73]
                t = np.linspace(0, 1, 100)
                control_left = [-50, 135]
                control_right = [50, 135]
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

            # 시각화 생성
            fig = go.Figure()

            # 내야 다이아몬드 추가
            fig.add_trace(go.Scatter(
                x=[x[0] for x in infield_diamond],
                y=[x[1] for x in infield_diamond],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

            # 외야 곡선 추가
            fig.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
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

            # 타구 데이터 추가
            for result, color in hit_color_mapping.items():
                subset = filtered_hits[filtered_hits['타격결과'] == result]
                fig.add_trace(go.Scatter(
                    x=subset['hit_X'],
                    y=subset['hit_Y'],
                    mode='markers',
                    marker=dict(color=color, size=8, opacity=0.8),
                    name=result
                ))

            # 레이아웃 설정
            fig.update_layout(
                title=f"{player_name} Spray Chart",
                xaxis=dict(title="X 좌표", range=[-90, 90], zeroline=False),
                yaxis=dict(title="Y 좌표", range=[0, 130], zeroline=False),
                showlegend=True,
                width=600, height=600,
                plot_bgcolor="white",
                margin=dict(l=50, r=50, t=50, b=50),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            return fig

        # 두 선수의 스프레이 차트를 양 옆으로 배치
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"### {pitcher1} Spray Chart")
            fig_1 = plot_spray_chart(filtered_hits_1, pitcher1)
            st.plotly_chart(fig_1)

        with col2:
            st.write(f"### {pitcher2} Spray Chart")
            fig_2 = plot_spray_chart(filtered_hits_2, pitcher2)
            st.plotly_chart(fig_2)
    else:
        st.warning("선수, 구종, 그리고 타격 결과를 모두 선택하면 시각화가 가능합니다.")
 
    # 데이터 필터링 및 시각화
    if pitcher1 and pitcher2 and pitch_type and selected_hit_results:
        filtered_hits_1 = df[
            (df['타자'] == pitcher1) &
            (df['구종'].isin(pitch_type)) &
            (df['타격결과'].isin(selected_hit_results))
        ]
        filtered_hits_2 = df[
            (df['타자'] == pitcher2) &
            (df['구종'].isin(pitch_type)) &
            (df['타격결과'].isin(selected_hit_results))
        ]

        # ------------------------
        # X-Z 시각화
        # ------------------------
        st.write("### 위에서 본 컨택 포인트")
        col1, col2 = st.columns(2)

        plate_x = [-21.59, -21.59, 0, 21.59, 21.59]
        plate_z = [43.18, 21.59, 0, 21.59, 43.18]

        def plot_xz(filtered_hits, player_name):
            fig = go.Figure()
            for result, color in hit_color_mapping.items():
                subset = filtered_hits[filtered_hits['타격결과'] == result]
                fig.add_trace(go.Scatter(
                    x=subset['ContactPositionZ'] * 100,
                    y=subset['ContactPositionX'] * 100,
                    mode='markers',
                    marker=dict(color=color, size=8, opacity=0.8),
                    name=result
                ))
            fig.add_trace(go.Scatter(
                x=plate_x, y=plate_z, mode='lines',
                line=dict(color='red', width=2), showlegend=False
            ))
            fig.update_layout(
                title=f"{player_name} X-Z 컨택 포인트",
                xaxis=dict(title="ContactPositionZ (좌우)", range=[-50, 50]),
                yaxis=dict(title="ContactPositionX (높이)", range=[-10, 100]),
                width=500, height=500, plot_bgcolor="white"
            )
            return fig

        with col1:
            st.plotly_chart(plot_xz(filtered_hits_1, pitcher1) if not filtered_hits_1.empty else st.warning(f"{pitcher1}의 데이터가 없습니다."))

        with col2:
            st.plotly_chart(plot_xz(filtered_hits_2, pitcher2) if not filtered_hits_2.empty else st.warning(f"{pitcher2}의 데이터가 없습니다."))

        # ------------------------
        # X-Y 시각화
        # ------------------------
        st.write("### 옆에서 본 컨택 포인트")
        col1, col2 = st.columns(2)

        plate_box_x = [0, 0, 43.18, 43.18, 0]
        plate_box_y = [46, 104, 104, 46, 46]

        def plot_xy(filtered_hits, player_name):
            fig = go.Figure()
            for result, color in hit_color_mapping.items():
                subset = filtered_hits[filtered_hits['타격결과'] == result]
                fig.add_trace(go.Scatter(
                    x=subset['ContactPositionX'] * 100,
                    y=subset['ContactPositionY'] * 100,
                    mode='markers',
                    marker=dict(color=color, size=8, opacity=0.8),
                    name=result
                ))
            fig.add_trace(go.Scatter(
                x=plate_box_x, y=plate_box_y, mode='lines',
                line=dict(color='red', width=2), showlegend=False
            ))
            fig.update_layout(
                title=f"{player_name} X-Y 컨택 포인트",
                xaxis=dict(title="ContactPositionX (좌우)", range=[-20, 120]),
                yaxis=dict(title="ContactPositionY (앞뒤)", range=[20, 120]),
                width=500, height=500, plot_bgcolor="white"
            )
            return fig

        with col1:
            st.plotly_chart(plot_xy(filtered_hits_1, pitcher1) if not filtered_hits_1.empty else st.warning(f"{pitcher1}의 데이터가 없습니다."))

        with col2:
            st.plotly_chart(plot_xy(filtered_hits_2, pitcher2) if not filtered_hits_2.empty else st.warning(f"{pitcher2}의 데이터가 없습니다."))

    else:
        st.warning("선수, 구종, 그리고 타격 결과를 모두 선택하면 시각화가 가능합니다.")

# -------------------
# Tab 2: 기간 간 비교
# -------------------
with tab2:
    st.subheader("기간 간 비교")

    # 선수 검색 및 선택
    search_query = st.text_input("선수 이름 검색", "").strip()
    if search_query:
        filtered_suggestions = [name for name in sorted(df['타자'].unique()) if search_query.lower() in name.lower()]
    else:
        filtered_suggestions = sorted(df['타자'].unique())

    if filtered_suggestions:
        # 리스트에 2개 이상의 항목이 있을 경우 두 번째 항목을 선택, 하나일 경우 첫 번째 선택
        if len(filtered_suggestions) >= 2:
            pitcher_name = st.selectbox("타자 이름 선택", filtered_suggestions, index=1, key="pitcher_search")
        else:
            pitcher_name = st.selectbox("타자 이름 선택", filtered_suggestions, index=0, key="pitcher_search")
    else:
        st.warning("검색된 선수가 없습니다.")
        pitcher_name = None

    # 기간 설정 (기간 1과 기간 2)
    col1, col2 = st.columns(2)
    with col1:
        start_date_1 = st.date_input("기간 1 시작 날짜", df['Date'].min(), key="start_date_1")
    with col2:
        end_date_1 = st.date_input("기간 1 종료 날짜", df['Date'].max(), key="end_date_1")

    col3, col4 = st.columns(2)
    with col3:
        start_date_2 = st.date_input("기간 2 시작 날짜", df['Date'].min(), key="start_date_2")
    with col4:
        end_date_2 = st.date_input("기간 2 종료 날짜", df['Date'].max(), key="end_date_2")

    # 데이터 필터링
    if pitcher_name:
        filtered_hits_1 = df[
            (df['타자'] == pitcher_name) &
            (df['Date'] >= pd.Timestamp(start_date_1)) &
            (df['Date'] <= pd.Timestamp(end_date_1))
        ]
        filtered_hits_2 = df[
            (df['타자'] == pitcher_name) &
            (df['Date'] >= pd.Timestamp(start_date_2)) &
            (df['Date'] <= pd.Timestamp(end_date_2))
        ]

        def plot_spray_chart(filtered_hits, player_name):
            # 빈 데이터프레임 확인
            if filtered_hits.empty:
                fig = go.Figure()
                fig.add_annotation(
                    text=f"{player_name}의 데이터가 없습니다.",
                    x=0.5, y=0.5, showarrow=False,
                    font=dict(size=16),
                    xref="paper", yref="paper"
                )
                fig.update_layout(
                    title=f"{player_name} Spray Chart",
                    xaxis=dict(title="X 좌표", range=[-90, 90], zeroline=False),
                    yaxis=dict(title="Y 좌표", range=[0, 130], zeroline=False),
                    width=600, height=600,
                    plot_bgcolor="white",
                    margin=dict(l=50, r=50, t=50, b=50),
                )
                return fig

            # 타구 위치 시각화 데이터 준비
            filtered_hits['hit_X'] = filtered_hits['hit_X'] * 1.0
            filtered_hits['hit_Y'] = filtered_hits['hit_Y'] * 1.0

            # 외야 곡선 생성
            def generate_outfield_curve():
                left_corner = [-73, 73]
                center = [0, 125]
                right_corner = [73, 73]
                t = np.linspace(0, 1, 100)
                control_left = [-50, 135]
                control_right = [50, 135]
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

            # 시각화 생성
            fig = go.Figure()

            # 내야 다이아몬드 추가
            fig.add_trace(go.Scatter(
                x=[x[0] for x in infield_diamond],
                y=[x[1] for x in infield_diamond],
                mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
            ))

            # 외야 곡선 추가
            fig.add_trace(go.Scatter(
                x=x_curve, y=y_curve, mode='lines',
                line=dict(color='black', width=2),
                showlegend=False
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

            # 타구 데이터 추가
            for result, color in hit_color_mapping.items():
                subset = filtered_hits[filtered_hits['타격결과'] == result]
                fig.add_trace(go.Scatter(
                    x=subset['hit_X'],
                    y=subset['hit_Y'],
                    mode='markers',
                    marker=dict(color=color, size=8, opacity=0.8),
                    name=result
                ))

            # 레이아웃 설정
            fig.update_layout(
                title=f"{player_name} Spray Chart",
                xaxis=dict(title="X 좌표", range=[-90, 90], zeroline=False),
                yaxis=dict(title="Y 좌표", range=[0, 130], zeroline=False),
                showlegend=True,
                width=600, height=600,
                plot_bgcolor="white",
                margin=dict(l=50, r=50, t=50, b=50),
            )
            fig.update_yaxes(scaleanchor="x", scaleratio=1)

            return fig

        # 두 선수의 스프레이 차트를 양 옆으로 배치
        col1, col2 = st.columns(2)

        with col1:
            st.write("### 기간 1 스프레이 차트")
            fig_1 = plot_spray_chart(filtered_hits_1, f"{pitcher_name} 기간 1 스프레이 차트")
            st.plotly_chart(fig_1, key="fig_period_1")

        with col2:
            st.write("### 기간 2 스프레이 차트")
            fig_2 = plot_spray_chart(filtered_hits_2, f"{pitcher_name} 기간 2 스프레이 차트")
            st.plotly_chart(fig_2, key="fig_period_2")

    # 데이터 필터링 및 시각화
    if pitcher1 and pitcher2 and pitch_type and selected_hit_results:
        filtered_hits_1 = df[
            (df['타자'] == pitcher_name) &
            (df['Date'] >= pd.Timestamp(start_date_1)) &
            (df['Date'] <= pd.Timestamp(end_date_1))
        ]
        filtered_hits_2 = df[
            (df['타자'] == pitcher_name) &
            (df['Date'] >= pd.Timestamp(start_date_2)) &
            (df['Date'] <= pd.Timestamp(end_date_2))
        ]

        # ------------------------
        # X-Z 시각화
        # ------------------------
        st.write("### 위에서 본 컨택 포인트")
        col1, col2 = st.columns(2)

        plate_x = [-21.59, -21.59, 0, 21.59, 21.59]
        plate_z = [43.18, 21.59, 0, 21.59, 43.18]

        def plot_xz(filtered_hits, player_name):
            fig = go.Figure()
            for result, color in hit_color_mapping.items():
                subset = filtered_hits[filtered_hits['타격결과'] == result]
                fig.add_trace(go.Scatter(
                    x=subset['ContactPositionZ'] * 100,
                    y=subset['ContactPositionX'] * 100,
                    mode='markers',
                    marker=dict(color=color, size=8, opacity=0.8),
                    name=result
                ))
            fig.add_trace(go.Scatter(
                x=plate_x, y=plate_z, mode='lines',
                line=dict(color='red', width=2), showlegend=False
            ))
            fig.update_layout(
                title=f"{player_name} X-Z 컨택 포인트",
                xaxis=dict(title="ContactPositionZ (좌우)", range=[-50, 50]),
                yaxis=dict(title="ContactPositionX (높이)", range=[-10, 100]),
                width=500, height=500, plot_bgcolor="white"
            )
            return fig

        with col1:
            st.plotly_chart(plot_xz(filtered_hits_1, pitcher_name), key="fig_xz_period_1")

        with col2:
            st.plotly_chart(plot_xz(filtered_hits_2, pitcher_name), key="fig_xz_period_2")

        # ------------------------
        # X-Y 시각화
        # ------------------------
        st.write("### 옆에서 본 컨택 포인트")
        col1, col2 = st.columns(2)

        plate_box_x = [0, 0, 43.18, 43.18, 0]
        plate_box_y = [46, 104, 104, 46, 46]

        def plot_xy(filtered_hits, player_name):
            fig = go.Figure()
            for result, color in hit_color_mapping.items():
                subset = filtered_hits[filtered_hits['타격결과'] == result]
                fig.add_trace(go.Scatter(
                    x=subset['ContactPositionX'] * 100,
                    y=subset['ContactPositionY'] * 100,
                    mode='markers',
                    marker=dict(color=color, size=8, opacity=0.8),
                    name=result
                ))
            fig.add_trace(go.Scatter(
                x=plate_box_x, y=plate_box_y, mode='lines',
                line=dict(color='red', width=2), showlegend=False
            ))
            fig.update_layout(
                title=f"{player_name} X-Y 컨택 포인트",
                xaxis=dict(title="ContactPositionX (좌우)", range=[-20, 120]),
                yaxis=dict(title="ContactPositionY (앞뒤)", range=[20, 120]),
                width=500, height=500, plot_bgcolor="white"
            )
            return fig

        with col1:
            st.plotly_chart(plot_xy(filtered_hits_1, pitcher_name), key="fig_xy_period_1")

        with col2:
            st.plotly_chart(plot_xy(filtered_hits_2, pitcher_name), key="fig_xy_period_2")
    else:
        st.warning("선수, 구종, 그리고 타격 결과를 모두 선택하면 시각화가 가능합니다.")

    