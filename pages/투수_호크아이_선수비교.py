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

def map_variable_name(variable):
    mapping = {
        "구속": "RelSpeed",
        "회전수": "SpinRate",
        "회전효율": "회전효율",
        "회전축": "Tilt",
        "수직무브먼트": "InducedVertBreak",
        "수평무브먼트": "HorzBreak",
        "릴리스높이": "RelHeight",
        "릴리스사이드": "RelSide",
        "익스텐션": "Extension"
    }
    return mapping.get(variable, variable)

# -------------------------------
# Streamlit 페이지 설정
# -------------------------------
st.set_page_config(
    page_title="23-24 호크아이 데이터 선수간 비교",
    page_icon="⚾",
    layout="wide"
)

if "logged_in" not in st.session_state or not st.session_state.logged_in:
    st.error("로그인 후에 이 페이지를 이용할 수 있습니다.")
    st.stop()

# -------------------------------
# 제목
# -------------------------------
st.title("호크아이 데이터 선수 간 비교분석")

# -------------------------------
# 탭 구성
# -------------------------------
tab1, tab2 = st.tabs(["선수 간 비교", "기간 간 비교"])

# -------------------------------
# Tab 1: 선수 간 비교
# -------------------------------
with tab1:
    st.subheader("선수 간 비교")

    # 선수 1과 선수 2 선택
    col1, col2 = st.columns(2)
    with col1:
        search_query_1 = st.text_input("선수 1 검색", key="search_query_1").strip()
        suggestions_1 = [name for name in sorted(df['투수'].unique()) if search_query_1.lower() in name.lower()] if search_query_1 else sorted(df['투수'].unique())
        pitcher1 = st.selectbox("선수 1 선택", suggestions_1, key="pitcher1")

    with col2:
        search_query_2 = st.text_input("선수 2 검색", key="search_query_2").strip()
        suggestions_2 = [name for name in sorted(df['투수'].unique()) if search_query_2.lower() in name.lower()] if search_query_2 else sorted(df['투수'].unique())
        pitcher2 = st.selectbox("선수 2 선택", suggestions_2, key="pitcher2")

    if pitcher1 and pitcher2:
        # 구종 및 비교 변수 선택
        pitch_type = st.multiselect("구종 선택", df['구종'].unique(), key="pitch_type_1")
        compare_variables = ["구속", "회전수", "회전효율", "회전축", "수직무브먼트", "수평무브먼트", "릴리스높이", "릴리스사이드", "익스텐션"]
        selected_variables = st.multiselect("비교할 변수 선택", compare_variables, key="selected_variables_1", default=compare_variables)
        

        # 데이터 필터링
        pitcher1_data = df[df['투수'] == pitcher1]
        pitcher2_data = df[df['투수'] == pitcher2]

        if pitch_type:
            filtered_df = pd.concat([
                pitcher1_data[pitcher1_data['구종'].isin(pitch_type)],
                pitcher2_data[pitcher2_data['구종'].isin(pitch_type)]
            ])
        else:
            filtered_df = pd.concat([pitcher1_data, pitcher2_data])  # 구종 미선택 시 전체 데이터 사용

        # 변수 비교 결과 계산
        comparison_results = []
        for variable in compare_variables:
            df_variable = map_variable_name(variable)
            if df_variable == "Tilt":
                value1 = pitcher1_data[df_variable].mode().iloc[0] if not pitcher1_data[df_variable].mode().empty else "N/A"
                value2 = pitcher2_data[df_variable].mode().iloc[0] if not pitcher2_data[df_variable].mode().empty else "N/A"
                diff = "N/A"  # Tilt는 숫자가 아니므로 차이 계산 불가
            else:
                value1 = round(pitcher1_data[df_variable].mean(), 2)
                value2 = round(pitcher2_data[df_variable].mean(), 2)
                diff = round(value2 - value1, 2)  # 차이 계산

            comparison_results.append({"변수": variable, "선수 1 평균": value1, "선수 2 평균": value2, "차이": diff})

        # 비교 결과 표시
        comparison_df = pd.DataFrame(comparison_results)
        st.subheader("선수 간 변수 비교 결과")
        st.dataframe(comparison_df)

        # 막대 그래프 출력
        for variable in compare_variables:
            df_variable = map_variable_name(variable)
            if df_variable != "Tilt":  # Tilt는 비수치형 변수이므로 제외
                combined_df = pd.DataFrame({
                    "선수": [pitcher1, pitcher2],
                    "평균값": [
                        comparison_df.loc[comparison_df["변수"] == variable, "선수 1 평균"].values[0],
                        comparison_df.loc[comparison_df["변수"] == variable, "선수 2 평균"].values[0]
                    ]
                })
                fig = px.bar(
                    combined_df,
                    x="선수",
                    y="평균값",
                    title=f"{variable} 비교",
                    labels={"평균값": variable},
                    color="선수"
                )
                st.plotly_chart(fig)

        # 구종별 수평/수직 무브먼트 산점도
        st.subheader("구종별 수평/수직 무브먼트")

        # 평균값 계산
        avg_data = filtered_df.groupby(["구종", "투수"])[["HorzBreak", "InducedVertBreak"]].mean().reset_index()

        # 산점도 생성 (평균값만 표시)
        fig = px.scatter(
            avg_data,
            x="HorzBreak",
            y="InducedVertBreak",
            color="구종",
            symbol="투수",  # '선수' 대신 '투수'로 수정 (데이터프레임에 포함된 열)
            hover_data={"HorzBreak": True, "InducedVertBreak": True, "투수": True},  # Hover에 평균값 표시
            title="구종별 수평/수직 무브먼트 (평균값만 표시)",
            color_discrete_map=cols,
            category_orders={"구종": list(cols.keys())},
            labels={"HorzBreak": "수평 무브 (cm)", "InducedVertBreak": "수직 무브 (cm)"}
        )

        # 산점도 설정
        fig.update_traces(marker=dict(size=12))  # 점 크기 조정
        fig.update_layout(
            width=800,  # 정사각형 비율: 가로 크기
            height=800,  # 정사각형 비율: 세로 크기
            xaxis=dict(
                range=[-70, 70],
                linecolor="black",
                zeroline=True,
                zerolinecolor="black"
            ),
            yaxis=dict(
                range=[-70, 70],
                linecolor="black",
                zeroline=True,
                zerolinecolor="black"
            ),
        )

        # 기준선 추가
        fig.add_shape(type="line", x0=0, y0=-70, x1=0, y1=70, line=dict(color="black", width=2))
        fig.add_shape(type="line", x0=-70, y0=0, x1=70, y1=0, line=dict(color="black", width=2))

        # Plotly 차트 출력
        st.plotly_chart(fig)

# -------------------------------
# Tab 2: 기간 간 비교
# -------------------------------
with tab2:
    st.subheader("기간 간 비교")
    
    # 선수 검색 및 선택
    search_query = st.text_input("선수 이름 검색", key="search_query_period").strip()
    suggestions = [name for name in sorted(df['투수'].unique()) if search_query.lower() in name.lower()] if search_query else sorted(df['투수'].unique())
    pitcher_name = st.selectbox("선수 선택", suggestions, key="pitcher_period")

    if pitcher_name:
        # 기간 선택
        col1, col2 = st.columns(2)
        start_date_1 = col1.date_input("기간 1 시작", df['Date'].min(), key="start_date_1")
        end_date_1 = col2.date_input("기간 1 종료", df['Date'].max(), key="end_date_1")
        start_date_2 = col1.date_input("기간 2 시작", df['Date'].min(), key="start_date_2")
        end_date_2 = col2.date_input("기간 2 종료", df['Date'].max(), key="end_date_2")

        # 구종 및 비교 변수 선택
        pitch_types = st.multiselect("구종 선택", df['구종'].unique(), key="pitch_type_2")
        compare_variables = ["구속", "회전수", "회전효율", "회전축", "수직무브먼트", "수평무브먼트", "릴리스높이", "릴리스사이드", "익스텐션"]
        selected_variables = st.multiselect("비교할 변수 선택", compare_variables, key="selected_variables_2", default=compare_variables)

        if st.button("기간 비교 실행", key="compare_periods"):
            # 데이터 필터링
            df_1 = df[(df['투수'] == pitcher_name) & (df['Date'] >= pd.Timestamp(start_date_1)) & (df['Date'] <= pd.Timestamp(end_date_1))]
            df_2 = df[(df['투수'] == pitcher_name) & (df['Date'] >= pd.Timestamp(start_date_2)) & (df['Date'] <= pd.Timestamp(end_date_2))]

            if pitch_types:
                df_1 = df_1[df_1['구종'].isin(pitch_types)]
                df_2 = df_2[df_2['구종'].isin(pitch_types)]

            # 변수별 비교 결과 계산
            comparison_results = []
            for variable in compare_variables:
                df_variable = map_variable_name(variable)
                if df_variable == "Tilt":
                    value1 = df_1[df_variable].mode().iloc[0] if not df_1[df_variable].mode().empty else "N/A"
                    value2 = df_2[df_variable].mode().iloc[0] if not df_2[df_variable].mode().empty else "N/A"
                else:
                    value1 = round(df_1[df_variable].mean(), 2)
                    value2 = round(df_2[df_variable].mean(), 2)
                comparison_results.append({"변수": variable, "기간 1 평균": value1, "기간 2 평균": value2})

            # 비교 결과 출력
            comparison_results = []
            for variable in compare_variables:
                df_variable = map_variable_name(variable)
                if df_variable == "Tilt":
                    value1 = df_1[df_variable].mode().iloc[0] if not df_1[df_variable].mode().empty else "N/A"
                    value2 = df_2[df_variable].mode().iloc[0] if not df_2[df_variable].mode().empty else "N/A"
                    diff = "N/A"  # Tilt는 숫자가 아니므로 차이 계산 불가
                else:
                    value1 = round(df_1[df_variable].mean(), 2)
                    value2 = round(df_2[df_variable].mean(), 2)
                    diff = round(value2 - value1, 2)  # 차이 계산

                comparison_results.append({
                    "변수": variable,
                    "기간 1 평균": value1,
                    "기간 2 평균": value2,
                    "차이": diff
                })

            # 비교 결과 데이터프레임 생성 및 출력
            comparison_df = pd.DataFrame(comparison_results)
            st.subheader("기간 간 변수 비교 결과")
            st.dataframe(comparison_df)

            # 막대 그래프 생성
            for variable in compare_variables:
                df_variable = map_variable_name(variable)
                if df_variable != "Tilt":  # Tilt는 비수치형 변수이므로 제외
                    combined_df = pd.DataFrame({
                        "기간": ["기간 1", "기간 2"],
                        "평균값": [
                            comparison_df.loc[comparison_df["변수"] == variable, "기간 1 평균"].values[0],
                            comparison_df.loc[comparison_df["변수"] == variable, "기간 2 평균"].values[0]
                        ]
                    })

                    # 막대 그래프 출력
                    fig = px.bar(
                        combined_df,
                        x="기간",
                        y="평균값",
                        title=f"{variable} 기간 간 비교",
                        labels={"평균값": variable},
                        color="기간"
                    )
                    st.plotly_chart(fig)

            # 구종별 수평/수직 무브먼트 산점도
            st.subheader("구종별 수평/수직 무브먼트")

            # 각 기간의 평균값 계산
            avg_df_1 = df_1.groupby("구종")[["HorzBreak", "InducedVertBreak"]].mean().reset_index().assign(기간="기간 1")
            avg_df_2 = df_2.groupby("구종")[["HorzBreak", "InducedVertBreak"]].mean().reset_index().assign(기간="기간 2")

            # 투수 이름 추가
            avg_df_1["투수"] = pitcher_name
            avg_df_2["투수"] = pitcher_name

            # 데이터 합치기
            combined_avg_data = pd.concat([avg_df_1, avg_df_2])

            # 산점도 출력 (평균값만 표시)
            fig = px.scatter(
                combined_avg_data,
                x="HorzBreak",
                y="InducedVertBreak",
                color="구종",
                symbol="기간",
                hover_data=["투수", "기간", "HorzBreak", "InducedVertBreak"],  # Hover에 평균값 표시
                title="구종별 수평/수직 무브먼트 (평균값만 표시)",
                color_discrete_map=cols,
                category_orders={"구종": list(cols.keys())},
                labels={"HorzBreak": "수평 무브 (cm)", "InducedVertBreak": "수직 무브 (cm)"}
            )

            # 산점도 스타일 설정
            fig.update_traces(marker=dict(size=12))  # 점 크기 증가
            fig.update_layout(
                width=800,  # 정사각형 비율: 가로 크기
                height=800,  # 정사각형 비율: 세로 크기
                xaxis=dict(
                    range=[-70, 70],
                    linecolor="black",
                    zeroline=True,
                    zerolinecolor="black"
                ),
                yaxis=dict(
                    range=[-70, 70],
                    linecolor="black",
                    zeroline=True,
                    zerolinecolor="black"
                ),
            )

            # 기준선 추가
            fig.add_shape(type="line", x0=0, y0=-70, x1=0, y1=70, line=dict(color="black", width=2))
            fig.add_shape(type="line", x0=-70, y0=0, x1=70, y1=0, line=dict(color="black", width=2))

            # Plotly 차트 출력
            st.plotly_chart(fig)