import streamlit as st

st.set_page_config(
    page_title="KIA 투수 데이터 분석",
    page_icon="🐯", 
    layout="wide"  # 전체 화면 사용
)


# 이미지 URL (GitHub Raw URL)
image_url = "https://raw.githubusercontent.com/JUNG-PFe/pitcher-visualization_2/main/wordmark.jpg"

# -------------------------------
# 로그인 상태 관리
# -------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False  # 초기값: 로그아웃 상태

# -------------------------------
# 아이디와 비밀번호 입력
# -------------------------------
st.sidebar.header("로그인 / 로그아웃")

# 사전에 설정된 유효한 아이디와 비밀번호
VALID_USERNAME = "KIA"
VALID_PASSWORD = "kiatigers11"

# 로그인 상태 확인
if not st.session_state.logged_in:
    # 로그인 폼
    with st.sidebar.form("login_form"):
        username = st.text_input("아이디")
        password = st.text_input("비밀번호", type="password")
        login_button = st.form_submit_button("로그인")

    # 로그인 검증
    if login_button:
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state.logged_in = True  # 로그인 성공 상태 저장
            st.sidebar.success(f"로그인 성공: {username}님 환영합니다!")
            
        else:
            st.sidebar.error("아이디 또는 비밀번호가 잘못되었습니다.")
else:
    st.sidebar.success(f"이미 로그인된 상태입니다.")
    # 로그아웃 버튼 추가
    if st.sidebar.button("로그아웃"):
        st.session_state.logged_in = False
        
        

# -------------------------------
# 페이지 활성화
# -------------------------------
if st.session_state.logged_in:
    # 상단 메인 타이틀
    st.title("KIA 투수 데이터 분석")
    st.write("왼쪽 메뉴에서 분석 페이지를 선택하세요.")

    # 상단 레이아웃: 로고는 왼쪽 상단에 배치
    col1, col2 = st.columns([1, 4])  # 비율을 조정해 로고를 왼쪽 정렬
    with col1:
        st.image(image_url, width=200)  # 로고 크기를 150x50으로 유지

    # 타이틀: 중앙에 정렬
    st.markdown(
        """
        <h1 style="text-align: center; margin-top: 10px; font-size: 4em;">KIA 투수 데이터 분석</h1>
        """,
        unsafe_allow_html=True,
    )

    # 설명 텍스트: 한 칸 아래로 내리고 글자 크기를 2배로 설정
    st.markdown(
        """
        <p style="text-align: center; font-size: 1.5em; margin-top: 20px;">사이드에서 분석 페이지를 선택하세요.</p>
        """,
        unsafe_allow_html=True,
    )

    # 페이지 하단 추가 내용
    st.write("---")
    st.write("1.PTS 데이터 24 : 24시즌 구속, 스크라이트, 볼 비율 등과 투구 로케이션 확인")
    st.write("2.PTS 데이터 존별 타구속도 24 : 24시즌 투수 존별 타구 속도 확인")
    st.write("3.호크아이 데이터 23-24 : 기본 트래킹 데이터 확인, 구종별 로케이션, 무브먼트 차트 확인")
    st.write("4.호크아이 피칭궤적 : 피칭궤적")
    st.write("5.호크아이 선수비교 : 선수간 트래킹 데이터 비교")
    st.write("6.호크아이 트렌드 분석 : 각 선수 트래킹 데이터 추이 분석")
else:
    st.markdown(
        """
        <h2 style="text-align: center; margin-top: 50px;">로그인 후에 사용 가능합니다.</h2>
        """,
        unsafe_allow_html=True,
    )