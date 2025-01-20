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
def load_data():
    # 데이터 URL
    data_url1 = "https://github.com/JUNG-PFe/pitcher-visualization_2/raw/refs/heads/main/24_merged_data_%EC%88%98%EC%A0%95.xlsx"
    data_url2 = "https://github.com/JUNG-PFe/pitcher-visualization_2/raw/refs/heads/main/23_merged_data_%EC%88%98%EC%A0%95.xlsx"
    
    # 데이터 로드
    df1 = pd.read_excel(data_url1)
    df2 = pd.read_excel(data_url2)
    
    # 날짜 형식 통일
    df1['Date'] = pd.to_datetime(df1['Date'])
    df2['Date'] = pd.to_datetime(df2['Date'])
    
    # 병합
    combined_df = pd.concat([df1, df2], ignore_index=True)
    return combined_df

# 데이터 로드
df = load_data()


st.set_page_config(
    page_title="23-24 호크아이 투수 데이터 필터링 및 분석 앱",
    page_icon="⚾",
    layout="wide"
)

# -------------------------------
# 로그인 여부 확인
# -------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("로그인 후에 이 페이지를 이용할 수 있습니다.")
    st.stop()  # 로그인 상태가 아니면 여기서 실행 중지

st.title("호크아이 데이터 23-24 분석")

# 세션 상태 초기화
if "filter_applied" not in st.session_state:
    st.session_state.filter_applied = False

# 데이터 필터링 섹션
st.subheader("데이터 필터링")

# -------------------
# 연도 및 달 필터 가로 배치
# -------------------
st.subheader("연도 및 달 ")
col1, col2 = st.columns(2)
with col1:
    unique_years = ["전체"] + sorted(df['Date'].dt.year.unique())
    selected_year = st.selectbox("연도 선택", unique_years)
with col2:
    unique_months = ["전체"] + list(range(1, 13))
    selected_month = st.selectbox("월 선택", unique_months)

# -------------------
# 날짜 범위 필터 (길게 표시)
# -------------------
st.subheader("기간 선택")
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
st.subheader("투수 이름")
col3, col4 = st.columns(2)
with col3:
    search_query = st.text_input("투수 이름 검색", "").strip()
    if search_query:
        suggestions = [name for name in sorted(df['투수'].unique()) if search_query.lower() in name.lower()]
    else:
        suggestions = sorted(df['투수'].unique())
with col4:
    if suggestions:
        pitcher_name = st.selectbox("투수 이름 선택", suggestions)
    else:
        pitcher_name = None

# -------------------
# 타자 유형 및 주자 상황 필터 가로 배치
# -------------------
st.subheader("타자 유형 및 주자 상황")
col5, col6 = st.columns(2)
with col5:
    batter_type = st.selectbox("타자 유형 선택", ["전체", "우타", "좌타"])
with col6:
    runner_status = st.selectbox("주자 상황 선택", ["전체", "주자무", "나머지"])

# -------------------
# 구종 및 타격결과 필터 가로 배치
# -------------------
st.subheader("구종 및 타격결과")
col7, col8 = st.columns(2)
with col7:
    pitch_type = st.multiselect("구종 선택", df['구종'].unique())
with col8:
    unique_hit_results = sorted(df['타격결과'].dropna().astype(str).unique())  # 고유한 타격결과 값 가져오기
    selected_hit_results = st.multiselect("타격결과 선택", ["전체"] + unique_hit_results, default=[])

# 검색 버튼
if st.button("검색 실행"):
    st.session_state.filter_applied = True

# 필터 적용 로직
if st.session_state.filter_applied:
    filtered_df = df.copy()

    # 날짜 필터 적용
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[(filtered_df['Date'] >= pd.Timestamp(start_date)) & (filtered_df['Date'] <= pd.Timestamp(end_date))]
    
    if selected_year != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.year == selected_year]
    if selected_month != "전체":
        filtered_df = filtered_df[filtered_df['Date'].dt.month == selected_month]

    # 투수 이름 필터 적용
    if pitcher_name:
        filtered_df = filtered_df[filtered_df['투수'] == pitcher_name]

    # 타자 유형 필터 적용
    if batter_type == "우타":
        filtered_df = filtered_df[filtered_df['타자유형'] == "우타"]
    elif batter_type == "좌타":
        filtered_df = filtered_df[filtered_df['타자유형'] == "좌타"]

    # 구종 필터 적용
    if pitch_type:
        filtered_df = filtered_df[filtered_df['구종'].isin(pitch_type)]

    # 주자 상황 필터 적용
    if runner_status == "주자무":
        filtered_df = filtered_df[filtered_df['주자'] == "주자무"]
    elif runner_status == "나머지":
        filtered_df = filtered_df[filtered_df['주자'] != "주자무"]

    # 타격결과 필터 적용
    if selected_hit_results:
        filtered_df = filtered_df[filtered_df['타격결과'].astype(str).isin(selected_hit_results)]

    # 기본 분석
    if not filtered_df.empty:
        st.subheader("기본 분석 값")
        analysis = filtered_df.groupby('구종').agg(
            투구수=('구종', 'count'),
            투구_비율=('구종', lambda x: round((x.count() / len(filtered_df)) * 100, 1)),
            스트라이크_비율=('심판콜', lambda x: round((x[x != 'B'].count() / x.count()) * 100, 1) if x.count() > 0 else 0),
            구속_평균=('RelSpeed', lambda x: round(x.mean(), 0)),
            구속_최고=('RelSpeed', lambda x: round(x.max(), 0)),
            회전수=('SpinRate', lambda x: round(x.mean(), 0)),
            회전효율=('회전효율', lambda x: round(x.mean(), 0)),
            Tilt=('Tilt', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
            수직무브_평균=('InducedVertBreak', lambda x: round(x.mean(), 1)),
            수평무브_평균=('HorzBreak', lambda x: round(x.mean(), 1)),
            타구속도=('ExitSpeed', lambda x: round(x.mean(), 0)),
            높이=('RelHeight', lambda x: round(x.mean() * 100, 0)),
            사이드=('RelSide', lambda x: round(x.mean() * 100, 0)),
            익스텐션=('Extension', lambda x: round(x.mean() * 100, 0))
        ).reset_index()

        analysis['구종'] = pd.Categorical(analysis['구종'], categories=list(cols.keys()), ordered=True)
        analysis = analysis.sort_values('구종')

        st.dataframe(analysis)

        # 구종별 플레이트 위치 시각화
        st.subheader("구종별 플레이트 위치")
        if "PlateLocSide" in filtered_df.columns and "PlateLocHeight" in filtered_df.columns:
            filtered_df['PlateLocSide'] = filtered_df['PlateLocSide'] * 100
            filtered_df['PlateLocHeight'] = filtered_df['PlateLocHeight'] * 100



        fig = px.scatter(
            filtered_df,
            x="PlateLocSide",
            y="PlateLocHeight",
            color="구종",
            title="구종별 플레이트 위치",
            color_discrete_map=cols,
            category_orders={"구종": list(cols.keys())},
            labels={"PlateLocSide": "좌우 위치 (cm)", "PlateLocHeight": "상하 위치 (cm)"}
        )
        fig.update_traces(marker=dict(size=15))
        fig.update_layout(
            width=700,  # 가로 크기
            height=800,  # 세로 크기
            xaxis=dict(range=[-70, 70], showline=False),
            yaxis=dict(range=[-10, 150], showline=False)
        )
        fig.add_shape(
            type="rect",
            x0=-23, x1=23,
            y0=46, y1=105,
            line=dict(color="gray", width=2),
            fillcolor="lightgray", opacity=0.2
        )
        st.plotly_chart(fig)

        # 구종별 수평/수직 무브먼트 시각화
        st.subheader("구종별 수평/수직 무브먼트")
        fig = px.scatter(
            filtered_df,
            x="HorzBreak",
            y="InducedVertBreak",
            color="구종",
            hover_data=["투수", "구속"],
            title="구종별 수평/수직 무브먼트",
            color_discrete_map=cols,
            category_orders={"구종": list(cols.keys())},
            labels={"HorzBreak": "수평 무브 (cm)", "InducedVertBreak": "수직 무브 (cm)"}
        )
        fig.update_traces(marker=dict(size=9))
        fig.update_layout(
            xaxis=dict(range=[-70, 70], linecolor="black"),
            yaxis=dict(range=[-70, 70], linecolor="black")
        )
        fig.update_layout(
            width=800,  # 가로 크기
            height=750,  # 세로 크기
            xaxis=dict(range=[-70, 70], linecolor="black"),
            yaxis=dict(range=[-70, 70], linecolor="black")
)
        fig.add_shape(type="line", x0=0, y0=-70, x1=0, y1=70, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-70, y0=0, x1=70, y1=0, line=dict(color="black", width=2))
        st.plotly_chart(fig)

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