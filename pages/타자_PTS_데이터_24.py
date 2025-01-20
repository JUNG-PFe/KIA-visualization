import pandas as pd
import streamlit as st
import plotly.express as px
import io
import numpy as np


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
    data_url = "https://github.com/JUNG-PFe/Batter_visualization/raw/refs/heads/main/PTS%202024%20%EC%A0%84%EA%B2%BD%EA%B8%B0_%EC%88%98%EC%A0%95_%ED%83%80%EA%B2%A9.xlsx"
    df = pd.read_excel(data_url)
    return df

df = load_new_data()

df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

# 앱 제목

st.set_page_config(
    page_title="24 PTS 데이터 필터링 및 분석 앱",
    page_icon="⚾",
    layout="wide"
)

# 로그인 여부 확인
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("로그인 후에 이 페이지를 이용할 수 있습니다.")
    st.stop()  # 로그인 상태가 아닌 경우 실행 중지

# 로그인 상태일 때만 아래 코드 실행
st.title("24 PTS 타자 데이터 필터링 및 존별 분석 앱")



# 세션 상태 초기화
if "filter_applied" not in st.session_state:
    st.session_state.filter_applied = False

# 데이터 필터링 섹션
st.subheader("데이터 필터링")

# -------------------
# 연도 및 달 필터 가로 배치
# -------------------
st.subheader("연도 및 달 필터")
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
# 투수 및 타자 검색 및 유형 필터 (3x2 레이아웃)
# -------------------
st.subheader("투수 및 타자 검색 및 유형 필터")
col1, col2 = st.columns(2)

# 첫 번째 열 (투수 유형, 투수 이름 검색, 투수 이름 선택)
with col1:
    # 투수 유형
    pitcher_throws = ["전체"] + sorted(df['PitcherThrows'].dropna().unique())
    selected_pitcher_throw = st.selectbox("투수 유형 선택 (좌투/우투)", pitcher_throws)

    # 투수 이름 검색
    pitcher_search_query = st.text_input("투수 이름 검색", "").strip()
    if pitcher_search_query:
        pitcher_suggestions = [name for name in sorted(df['Pitcher'].unique()) if pitcher_search_query.lower() in name.lower()]
    else:
        pitcher_suggestions = sorted(df['Pitcher'].unique())

    # 투수 이름 선택
    if pitcher_suggestions:
        pitcher_name = st.selectbox("투수 이름 선택", ["전체"]+ pitcher_suggestions)
    else:
        pitcher_name = None

# 두 번째 열 (타자 유형, 타자 이름 검색, 타자 이름 선택)
with col2:
    # 타자 유형
    batter_sides = ["전체"] + sorted(df['BatterSide'].dropna().unique())
    selected_batter_side = st.selectbox("타자 유형 선택 (좌타/우타)", batter_sides)

    # 타자 이름 검색
    batter_search_query = st.text_input("타자 이름 검색", "").strip()
    if batter_search_query:
        batter_suggestions = [name for name in sorted(df['Batter'].unique()) if batter_search_query.lower() in name.lower()]
    else:
        batter_suggestions = sorted(df['Batter'].unique())

    # 타자 이름 선택
    if batter_suggestions:
        Batter_name = st.selectbox("타자 이름 선택", ["전체"] + batter_suggestions)
    else:
        Batter_name = "전체"


# -------------------
# 주자 상황 및 볼 카운트 필터 가로 배치
# -------------------
st.subheader("주자 상황 및 볼 카운트")
col5, col6 = st.columns(2)
with col5:
    runner_status = st.selectbox("주자 상황 선택", ["전체", "주자무", "나머지"])
with col6:
    unique_bcounts = ["전체"] + sorted(df['BCOUNT'].unique())
    selected_bcount = st.selectbox("볼카운트 선택", unique_bcounts)

# -------------------
# 구종 및 타격결과 필터 가로 배치
# -------------------
st.subheader("구종 및 타격결과")
col7, col8 = st.columns(2)
with col7:
    pitch_type = st.multiselect("구종 선택", df['PitchType'].unique())
with col8:
    unique_hit_results = sorted(df['Result'].dropna().astype(str).unique())
    selected_hit_results = st.multiselect("타격결과 선택", ["전체"] + unique_hit_results, default=[])

# 검색 버튼
if st.button("검색 실행"):
    st.session_state.filter_applied = True



if st.session_state.filter_applied:
    # 필터링된 데이터 사용
    filtered_df = df.copy()

    # 연도 및 달 필터 적용
    if selected_year != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.year == int(selected_year)]
    if selected_month != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.month == int(selected_month)]

    # 날짜 범위 필터 적용
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.Timestamp(start_date)) & (filtered_df['Date'] <= pd.Timestamp(end_date))]

    # 투수 유형 및 이름 필터 적용
    if selected_pitcher_throw != "전체":
        filtered_df = filtered_df[filtered_df['PitcherThrows'] == selected_pitcher_throw]
    if pitcher_name != "전체" and pitcher_name is not None:
        filtered_df = filtered_df[filtered_df['Pitcher'] == pitcher_name]

    # 타자 유형 및 이름 필터 적용
    if selected_batter_side != "전체":
        filtered_df = filtered_df[filtered_df['BatterSide'] == selected_batter_side]
    if Batter_name != "전체":
        filtered_df = filtered_df[filtered_df['Batter'] == Batter_name]

    # 주자 상황 필터 적용
    if runner_status != "전체":
        if runner_status == "주자무":
            filtered_df = filtered_df[filtered_df['RunnerStatus'] == "주자무"]
        elif runner_status == "나머지":
            filtered_df = filtered_df[filtered_df['RunnerStatus'] != "주자무"]

    # 볼카운트 필터 적용
    if selected_bcount != "전체":
        filtered_df = filtered_df[filtered_df['BCOUNT'] == selected_bcount]

    # 구종 필터 적용
    if pitch_type:
        filtered_df = filtered_df[filtered_df['PitchType'].isin(pitch_type)]

    # 타격결과 필터 적용
    if selected_hit_results and "전체" not in selected_hit_results:
        filtered_df = filtered_df[filtered_df['Result'].isin(selected_hit_results)]

    filtered_df['PTS_location_X'] = pd.to_numeric(filtered_df['PTS_location_X'], errors='coerce')
    filtered_df['PTS_location_Z'] = pd.to_numeric(filtered_df['PTS_location_Z'], errors='coerce')
    filtered_df.dropna(subset=['PTS_location_X', 'PTS_location_Z'], inplace=True)

    # 스트라이크 존 경계 설정
    strike_zone_x_edges = np.linspace(-23, 23, 4)
    strike_zone_z_edges = np.linspace(46, 105, 4)

    # 존 설정
    filtered_df['Zone'] = pd.cut(
        filtered_df['PTS_location_X'], bins=strike_zone_x_edges, labels=[3, 2, 1]  # X축 좌→우 순서 변경
    ).astype(str) + pd.cut(
        filtered_df['PTS_location_Z'], bins=strike_zone_z_edges, labels=[3, 2, 1]  # Y축 낮음→높음 순서 변경
    ).astype(str)

    # 분석 지표 선택 및 계산
    metric = st.selectbox("분석 지표 선택", ['인플레이 타구', '파울', '스윙중 헛스윙', '히트', '장타', '뜬공', '땅볼'])
    metric_mapping = {'인플레이 타구': 'H', '파울': 'F', '스윙중 헛스윙': 'S'}
    metric = metric_mapping.get(metric, metric)

    # '히트', '장타', '땅볼', '뜬공' 정의
    if metric == '히트':
        filtered_df['히트'] = filtered_df['Result'].apply(
            lambda x: x in ['안타', '2루타', '3루타', '홈런']
        )
    elif metric == '장타':
        filtered_df['장타'] = filtered_df['Result'].apply(
            lambda x: x in ['2루타', '3루타', '홈런']
        )
    elif metric == '땅볼':
        filtered_df['땅볼'] = filtered_df['Result'].apply(
            lambda x: x == '땅볼'
        )
    elif metric == '뜬공':
        filtered_df['뜬공'] = filtered_df['Result'].apply(
            lambda x: x == '뜬공'
        )

    if metric in ['H', 'F', 'S']:
        # 'PitchCall' 데이터를 사용하는 경우
        zone_summary = (
            filtered_df.groupby('Zone')['PitchCall']
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        zone_summary = zone_summary.reindex(columns=['F', 'H', 'S'], fill_value=0)
        selected_rate = zone_summary[metric]
    elif metric in ['히트', '장타', '땅볼', '뜬공']:
        # '히트', '장타', '땅볼', '뜬공' 비율 계산
        selected_rate = filtered_df.groupby('Zone')[metric].mean()
    else:
        # 'Result' 데이터를 사용하는 경우
        selected_rate = (
            filtered_df.groupby('Zone')['Result']
            .apply(lambda x: (x == metric).mean())
        )

    # Zone별 전체 Pitch 수
    zone_counts = filtered_df.groupby('Zone')['PitchCall'].size()

    # Zone별 성공 횟수 계산
    if metric in ['H', 'F', 'S']:
        zone_success = filtered_df.groupby('Zone')['PitchCall'].apply(lambda x: (x == metric).sum())
    elif metric in ['히트', '장타', '땅볼', '뜬공']:
        zone_success = filtered_df.groupby('Zone')[metric].sum()
    else:
        zone_success = filtered_df.groupby('Zone')['Result'].apply(lambda x: (x == metric).sum())

    # Zone 값이 누락된 경우 0으로 채우기
    zone_labels = ['33', '32', '31', '23', '22', '21', '13', '12', '11']
    selected_rate = selected_rate.reindex(zone_labels, fill_value=0)
    zone_counts = zone_counts.reindex(zone_labels, fill_value=0)
    zone_success = zone_success.reindex(zone_labels, fill_value=0)

    # 3x3 행렬 변환
    zone_matrix = selected_rate.values.reshape(3, 3)
    zone_matrix_percent = (zone_matrix * 100).round(1)
    zone_matrix_counts = zone_counts.values.reshape(3, 3)
    zone_matrix_success = zone_success.values.reshape(3, 3)

    # Plotly 히트맵 생성
    fig = px.imshow(
        zone_matrix,
        labels=dict(x="좌/우", y="높음/낮음", color=f"{metric} 비율 (%)"),
        x=['우', '중', '좌'],  # X축 이름
        y=['높음', '중간', '낮음'],  # Y축 이름
        color_continuous_scale="Reds",
        zmin=zone_matrix.min(),
        zmax=zone_matrix.max(),
        text_auto=False
    )

    # Y축 순서 명시
    fig.update_yaxes(
        categoryarray=['높음', '중간', '낮음'],
        categoryorder='array'
    )

    # X축 순서 명시
    fig.update_xaxes(
        categoryarray=['우', '중', '좌'],
        categoryorder='array'
    )

    # 텍스트 추가
    min_val = np.nanmin(zone_matrix)
    max_val = np.nanmax(zone_matrix)
    threshold = min_val + (max_val - min_val) * 0.6

    for i in range(zone_matrix_percent.shape[0]):
        for j in range(zone_matrix_percent.shape[1]):
            success = int(zone_matrix_success[i, j])
            total = int(zone_matrix_counts[i, j])
            percentage = zone_matrix_percent[i, j]

            text_color = "white" if zone_matrix[i, j] > threshold else "black"

            fig.add_annotation(
                text=f"{percentage}%<br>({success}/{total})",
                x=j,
                y=i,
                showarrow=False,
                font=dict(color=text_color, size=12)
            )

    # 레이아웃 설정
    fig.update_layout(
        title=f"스트라이크 존별 {metric} 비율_포수시점",
        xaxis_title="좌/우",
        yaxis_title="높음/낮음",
        width=800,
        height=800
    )

    # Streamlit에서 그래프 표시
    st.plotly_chart(fig, use_container_width=True)

    

    # 선택된 지표에 따라 필터링된 데이터 사용
    selected_result = st.selectbox(
        "타격 결과 선택",
        ['전체', '안타', '2루타', '3루타', '홈런', '뜬공', '땅볼'],
        index=0
    )

    # 선택된 결과에 따라 필터링
    if selected_result != "전체":
        filtered_df_2d = filtered_df[filtered_df['Result'] == selected_result]
    else:
        filtered_df_2d = filtered_df[filtered_df['Result'].isin(['안타', '2루타', '3루타', '홈런', '뜬공', '땅볼'])]

    fig_2d = px.scatter(
        filtered_df_2d,
        x='PTS_ExitSpeed',  # X축
        y='PTS_Angle',      # Y축
        color='Result',     # 색상을 Result로 구분
        title=f"2D 산점도: 타구 속도와 각도 ({selected_result})",
        labels={
            'PTS_ExitSpeed': '타구 속도',
            'PTS_Angle': '타구 각도'
        },
        opacity=0.7  # 투명도 설정
    )

    # 레이아웃 설정
    fig_2d.update_layout(
        xaxis=dict(
            title='타구 속도',
            title_font=dict(size=14, color='black'),  # X축 제목 스타일
            showline=True,
            linewidth=2,  # X축 선 두께
            linecolor='black',  # X축 선 색상
            mirror=False, 
            range=[20, 190]
        ),
        yaxis=dict(
            title='타구 각도',
            title_font=dict(size=14, color='black'),  # Y축 제목 스타일
            showline=True,
            linewidth=2,  # Y축 선 두께
            linecolor='black',  # Y축 선 색상
            mirror=False,
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
        width=800,  # 원하는 폭 (px 단위)
        height=900  # 원하는 높이 (px 단위)
    )

    # Streamlit에서 그래프 표시
    st.plotly_chart(fig_2d, use_container_width=True)