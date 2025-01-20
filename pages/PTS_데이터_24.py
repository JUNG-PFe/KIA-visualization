import pandas as pd
import streamlit as st
import plotly.express as px
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
st.title("24 PTS 투수 데이터 필터링 및 분석 앱")



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
    if pitcher_name != "전체":  # 투수 이름이 선택된 경우에만 필터 적용
        filtered_df = filtered_df[filtered_df['Pitcher'] == pitcher_name]

    # 타자 이름 필터
    if Batter_name != "전체":
        filtered_df = filtered_df[filtered_df['Batter'] == Batter_name]

    # 구종 필터
    if pitch_type:
        filtered_df = filtered_df[filtered_df['PitchType'].isin(pitch_type)]

    # 주자 상황 필터
    if runner_status == "주자무":
        filtered_df = filtered_df[filtered_df['Runners'] == "주자무"]
    elif runner_status == "나머지":
        filtered_df = filtered_df[filtered_df['Runners'] != "주자무"]

    # 볼 카운트 필터
    if selected_bcount != "전체":
        filtered_df = filtered_df[filtered_df['BCOUNT'] == selected_bcount]
    # 타격결과 필터 적용
    if selected_hit_results:
        filtered_df = filtered_df[filtered_df['Result'].astype(str).isin(selected_hit_results)]

    # 타자 유형 필터
    if selected_batter_side != "전체":
        filtered_df = filtered_df[filtered_df['BatterSide'] == selected_batter_side]

    # 투수 유형 필터
    if selected_pitcher_throw != "전체":
        filtered_df = filtered_df[filtered_df['PitcherThrows'] == selected_pitcher_throw]

    # PTS 데이터 전처리
    filtered_df['PTS_location_X'] = pd.to_numeric(filtered_df['PTS_location_X'], errors='coerce')
    filtered_df['PTS_location_Z'] = pd.to_numeric(filtered_df['PTS_location_Z'], errors='coerce')
    filtered_df = filtered_df.dropna(subset=['PTS_location_X', 'PTS_location_Z'])

    # 기본 분석
    if not filtered_df.empty:
        st.subheader("기본 분석 값")
        analysis = filtered_df.groupby('PitchType').agg(
            투구수=('PitchType', 'count'),
            투구_비율=('PitchType', lambda x: round((x.count() / len(filtered_df)) * 100, 1)),
            스트라이크_비율=('PitchCall', lambda x: round((x[x != 'B'].count() / x.count()) * 100, 1) if x.count() > 0 else 0),
            구속_평균=('PTS_Speed', lambda x: round(x.mean(), 0)),
            구속_최고=('PTS_Speed', lambda x: round(x.max(), 0)),
            헛스윙S_비율=('PitchCall', lambda x: round((x[x == 'S'].count() / x.count()) * 100, 1) if x.count() > 0 else 0),  # S 비율
            루킹S_비율=('PitchCall', lambda x: round((x[x == 'T'].count() / x.count()) * 100, 1) if x.count() > 0 else 0),  # T 비율
            파울_비율=('PitchCall', lambda x: round((x[x == 'F'].count() / x.count()) * 100, 1) if x.count() > 0 else 0),  # F 비율
            안타_비율=('PitchCall', lambda x: round((x[x == 'H'].count() / x.count()) * 100, 1) if x.count() > 0 else 0) , 
            볼_비율=('PitchCall', lambda x: round((x[x == 'B'].count() / x.count()) * 100, 1) if x.count() > 0 else 0)  # B 비율
            
        ).reset_index()
        analysis['PitchType'] = pd.Categorical(analysis['PitchType'], categories=list(cols.keys()), ordered=True)
        analysis = analysis.sort_values('PitchType')

        st.dataframe(analysis)


    # 전체 데이터 산점도 생성
    if not filtered_df.empty:
        # 전체 데이터 투구 수 계산
        total_pitch_count = len(filtered_df)

        # 전체 데이터 산점도 생성
        fig_all = px.scatter(
            filtered_df,
            x="PTS_location_X",
            y="PTS_location_Z",
            color="PitchType",
            title=" ",
            color_discrete_map=cols,  # 색상 매핑
            category_orders={"PitchType": list(cols.keys())},
            labels={"PTS_location_X": "좌우 위치 (cm)", "PTS_location_Z": "상하 위치 (cm)"},
            opacity=0.7,  # 점의 투명도 설정
        )
        fig_all.update_traces(marker=dict(size=10))

        # 산점도 레이아웃 조정
        fig_all.update_layout(
            width=800,  # 산점도 너비
            height=750,  # 산점도 높이
            margin=dict(l=10, r=10, t=30, b=10),  # 여백 조정
            xaxis=dict(
                range=[-70, 70],  # X축 범위 고정
                showgrid=False,
                zeroline=False,
                visible=True,
                title="PTS Location X"
            ),
            yaxis=dict(
                range=[-10, 150],  # Y축 범위 고정
                showgrid=False,
                zeroline=False,
                visible=True,
                title="PTS Location Z"
            )
        )

        # 스트라이크 존 추가
        fig_all.add_shape(
            type="rect",
            x0=-23, x1=23,
            y0=46, y1=105,
            line=dict(color="gray", width=2),
            fillcolor="lightgray", opacity=0.2
        )

        # 전체 구종 산점도 출력
        st.subheader("전 구종 로케이션_포수시점")
        st.plotly_chart(fig_all)
        st.markdown(f"**{total_pitch_count} Pitches**")
        st.markdown("---")  # 구분선 추가

    # 구종별 데이터 추출
    pitch_types = filtered_df['PitchType'].unique()

    # 구종별 산점도 생성
    for pitch_type in pitch_types:
        # 구종별 데이터 필터링
        filtered_data = filtered_df[filtered_df['PitchType'] == pitch_type]

        # 구종별 투구 수 및 비율 계산
        pitch_count = len(filtered_data)
        pitch_percentage = (pitch_count / len(filtered_df)) * 100  # 필터링된 데이터 기준으로 비율 계산

        # 산점도 생성
        fig = px.scatter(
            filtered_data,
            x="PTS_location_X",
            y="PTS_location_Z",
            color="PitchType",
            title="",
            color_discrete_map=cols,  # 색상 매핑
            category_orders={"PitchType": list(cols.keys())},
            labels={"PTS_location_X": "좌우 위치 (cm)", "PTS_location_Z": "상하 위치 (cm)"},
            opacity=0.7,  # 점의 투명도 설정
        )
        fig.update_traces(marker=dict(size=15))

        # 산점도 레이아웃 조정
        fig.update_layout(
            width=400,  # 산점도 너비
            height=400,  # 산점도 높이
            margin=dict(l=10, r=10, t=10, b=10),  # 여백 최소화
            xaxis=dict(
                range=[-70, 70],  # X축 범위 고정
                showgrid=False,
                zeroline=False,
                visible=False
            ),
            yaxis=dict(
                range=[0, 150],  # Y축 범위 고정
                showgrid=False,
                zeroline=False,
                visible=False
            )
        )

        # 스트라이크 존 추가
        fig.add_shape(
            type="rect",
            x0=-23, x1=23,
            y0=46, y1=105,
            line=dict(color="gray", width=2),
            fillcolor="lightgray", opacity=0.2
        )

        # 산점도 출력
        st.subheader(f"{pitch_type}")  # 구종 이름 표시
        st.plotly_chart(fig)

        # 구종 정보 표시
        st.markdown(f"**{pitch_count} Pitches** ({pitch_percentage:.1f}%)")
        st.markdown("---")  # 구분선 추가

    # 데이터 다운로드
    st.subheader("결과 다운로드")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        filtered_df.to_excel(writer, index=False, sheet_name='Filtered Data')
    output.seek(0)

    st.download_button(
        label="필터링된 데이터 다운로드 (Excel)",
        data=output,
        file_name='filtered_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )
else:
    st.info("필터링 조건에 맞는 데이터가 없습니다. 조건을 수정해주세요.")
