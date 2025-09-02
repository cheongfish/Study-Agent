import streamlit as st
import requests
import time

# FastAPI 서버 주소
FASTAPI_URL = "http://127.0.0.1:8000/generate"

# --- Streamlit UI 설정 ---
st.set_page_config(page_title="Study Agent Chatbot", page_icon="🤖")
st.title("🤖 Study Agent Chatbot")
st.caption("LangGraph RAG Agent와 대화하여 학습 콘텐츠를 생성해보세요.")

# 세션 상태에 메시지 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "안녕하세요! 어떤 주제에 대한 학습 콘텐츠를 만들어 드릴까요?"}
    ]

# 이전 대화 내용 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("예: 고등학교 1학년 수학 '집합'"):
    # 사용자 메시지를 세션 상태와 화면에 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 어시스턴트 응답을 위한 준비
    with st.chat_message("assistant"):
        # 로딩 스피너 표시
        with st.spinner("에이전트가 생각 중입니다... 잠시만 기다려주세요."):
            try:
                # FastAPI 서버로 요청 전송 (5분 타임아웃)
                response = requests.post(FASTAPI_URL, json={"prompt": prompt}, timeout=90)
                response.raise_for_status()  # HTTP 오류 발생 시 예외 처리

                data = response.json()
                full_response = data.get("response", "죄송합니다, 응답을 생성하는 데 실패했습니다.")

            except requests.exceptions.RequestException as e:
                full_response = f"서버 연결 오류: {e}\n\nFastAPI 서버(`app.py`)가 실행 중인지 확인해주세요."
                st.error(full_response)
            except Exception as e:
                full_response = f"알 수 없는 오류가 발생했습니다: {e}"
                st.error(full_response)

        # 타이핑 효과를 주며 응답 표시
        response_container = st.empty()
        typed_response = ""
        for chunk in full_response.split():
            typed_response += chunk + " "
            time.sleep(0.05)
            response_container.markdown(typed_response + "▌")
        response_container.markdown(typed_response)

    # 최종 응답을 세션 상태에 저장
    st.session_state.messages.append({"role": "assistant", "content": full_response})