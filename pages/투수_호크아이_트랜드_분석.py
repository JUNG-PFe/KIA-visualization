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
    page_title="23-24 호크아이 투수 데이터 트랜드 분석",
    page_icon="⚾",
    layout="wide"
)

# -------------------------------
# 로그인 여부 확인
# -------------------------------
if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("로그인 후에 이 페이지를 이용할 수 있습니다.")
    st.stop()  # 로그인 상태가 아니면 여기서 실행 중지

st.title("호크아이 데이터 트랜드 분석")

# 투수 이름 필터
search_query = st.text_input("투수 이름 검색", "").strip()
if search_query:
    suggestions = [name for name in sorted(df['투수'].unique()) if search_query.lower() in name.lower()]
else:
    suggestions = sorted(df['투수'].unique())

if suggestions:
    pitcher_name = st.selectbox("투수 이름 선택", suggestions)
else:
    pitcher_name = None

# 날짜 범위 선택
st.sidebar.header("날짜 범위 선택")
start_date = st.sidebar.date_input("시작 날짜", df['Date'].min())
end_date = st.sidebar.date_input("종료 날짜", df['Date'].max())

# 구종 필터
st.sidebar.header("구종 필터")
pitch_types = st.sidebar.multiselect(
    "구종을 선택하세요",
    options=sorted(df['구종'].unique()),  # 유일한 구종 목록
    default=sorted(df['구종'].unique())  # 기본값: 모든 구종
)

# 검색 버튼
if "filter_applied" not in st.session_state:
    st.session_state.filter_applied = False

if st.button("검색 실행"):
    st.session_state.filter_applied = True

# 데이터 필터링 및 시각화
if st.session_state.filter_applied:
    # 선택된 투수와 날짜에 따라 데이터 필터링
    filtered_df = df[(df['Date'] >= pd.Timestamp(start_date)) & (df['Date'] <= pd.Timestamp(end_date))]
    if pitcher_name:
        filtered_df = filtered_df[filtered_df['투수'] == pitcher_name]
    if pitch_types:
        filtered_df = filtered_df[filtered_df['구종'].isin(pitch_types)]

    if filtered_df.empty:
        st.warning("선택된 날짜 범위, 투수 또는 구종에 해당하는 데이터가 없습니다.")
    else:
        # 15일 간격으로 데이터 집계
        filtered_df['15_day_interval'] = filtered_df['Date'].dt.to_period('15D').apply(lambda r: r.start_time)
        
        # 집계 함수 설정
        aggregated_df = filtered_df.groupby(['15_day_interval', '구종']).agg({
            'RelSpeed': lambda x: round(x.mean(), 0),
            'SpinRate': lambda x: round(x.mean(), 0),
            '회전효율': lambda x: round(x.mean(), 0),
            'InducedVertBreak': lambda x: round(x.mean(), 1),
            'HorzBreak': lambda x: round(x.mean(), 1),
            'RelHeight': lambda x: round(x.mean() * 100, 0),
            'RelSide': lambda x: round(x.mean() * 100, 0),
            'Extension': lambda x: round(x.mean() * 100, 0)
        }).reset_index()

        # 시각화할 변수 선택 (다중 선택)
        st.sidebar.header("시각화할 변수 선택")
        selected_variables = st.sidebar.multiselect(
            "변수를 선택하세요",
            ["RelSpeed", "SpinRate", "회전효율", "InducedVertBreak", "HorzBreak", "RelHeight", "RelSide", "Extension"],
            default=["RelSpeed"]  # 기본값 설정
        )

        if selected_variables:
            for variable in selected_variables:
                # 트렌드 시각화
                fig = px.line(
                    aggregated_df,
                    x='15_day_interval',
                    y=variable,
                    color='구종',
                    color_discrete_map=cols,  # 색상 매핑 적용
                    title=f"{pitcher_name}의 15일 간격 투구 유형별 {variable} 트렌드" if pitcher_name else f"15일 간격 투구 유형별 {variable} 트렌드",
                    labels={"15_day_interval": "날짜", variable: variable, "구종": "구종"}
                )
                st.plotly_chart(fig)

        # 결과 다운로드
        st.subheader("결과 다운로드")
        output = io.BytesIO()

        # 필터링된 데이터를 Excel로 저장
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            aggregated_df.to_excel(writer, index=False, sheet_name='Aggregated Data')  # 집계된 데이터 저장
            filtered_df.to_excel(writer, index=False, sheet_name='Filtered Data')  # 필터링된 원본 데이터도 추가로 저장

        output.seek(0)

        # 다운로드 버튼 (Excel 파일)
        st.download_button(
            label="필터링된 데이터 다운로드 (Excel)",
            data=output,
            file_name='filtered_data.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
