from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from psycopg2.extensions import register_adapter
from psycopg2.extensions import AsIs
import uvicorn

from Compile_graph import get_compiled_graph



# LangGraph 애플리케이션 컴파일
app = get_compiled_graph()


# pgvector가 numpy.float32 타입을 인식할 수 있도록 어댑터 등록
register_adapter(np.float32, lambda a: AsIs(a.item()))

# FastAPI 애플리케이션 생성
server = FastAPI(
    title="Study Agent API",
    description="LangGraph RAG Agent를 이용한 학습 콘텐츠 생성 API",
    version="1.0.0",
)

# 요청 본문을 위한 Pydantic 모델 정의
class PromptRequest(BaseModel):
    prompt: str

@server.post("/generate", summary="학습 콘텐츠 생성", description="사용자 프롬프트에 기반하여 학습 목표와 연습 문제를 생성합니다.")
def generate_content(request: PromptRequest):
    """
    사용자로부터 프롬프트를 받아 LangGraph RAG 에이전트를 실행하고,
    생성된 학습 콘텐츠를 반환합니다.

    - **prompt**: 사용자가 입력한 학습 콘텐츠 생성 요청 문자열.
    """
    inputs = {"prompt": request.prompt}
    # LangGraph를 동기적으로 실행하고 최종 결과를 받습니다.
    final_state = app.invoke(inputs)
    # 최종 응답을 JSON 형태로 반환합니다.
    return {"response": final_state['final_response']}

# 서버 실행 (uvicorn)
if __name__ == "__main__":
    uvicorn.run(server, host="0.0.0.0", port=8000)