from pydantic import BaseModel,Field
from typing import List,Literal,Optional
from typing import TypedDict

# 노드 1의 출력 구조를 정의할 Pydantic 모델
class Requirements(BaseModel):
    """사용자 프롬프트에서 추출된 요구사항"""
    school_level: str = Field(description="학교급 (예: 초등학교, 중학교, 고등학교)")
    grade: str = Field(description="학년 (예: 1학년, 2학년)")
    subject: str = Field(description="과목 (예: 수학, 과학)")
    content_requests: List[Literal["학습 목표 생성", "문제 생성"]] = Field(description="요청된 콘텐츠 유형 목록")
    domain: str = Field(description="핵심 학습 주제")
    basecode: str

# 데이터 모델 정의: 검색된 문서의 관련성을 이진 점수로 평가하기 위한 데이터 모델
class GradeDocuments(BaseModel):
    """A binary score to determine the relevance of the retrieved documents."""

    # 문서가 질문에 관련이 있는지 여부를 'yes' 또는 'no'로 나타내는 필드
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )
class ProblemState(TypedDict):
    problems: Optional[str]               # 노드의 결과 (생성된 문제)
    status: str                           # 노드 실행 상태 (success/failure)
    error_message: Optional[str]          # 에러 메시지
    
class GoalState(TypedDict):
    learning_goals: Optional[str]         # 노드의 결과 (생성된 학습 목표)
    status: str                           # 노드 실행 상태 (success/failure)
    error_message: Optional[str]          # 에러 메시지
    
class MainState(TypedDict):
    prompt: str                          # 사용자 초기 입력
    metadata: str                        # 과목 메타데이터
    requirements: Optional[Requirements]  # 노드 1의 결과 (추출된 요구사항)
    retrieved_docs: Optional[List[dict]]  # 노드 2의 결과 (벡터 DB 검색 결과)
    binary_score: str                    # 관련성 검증 노드의 결과 값
    final_response: Optional[str]        # 최종 답변

    # 병렬 노드들의 상태를 여기에 포함 (개별 객체로 관리)
    problem_state: Optional[ProblemState]
    goal_state: Optional[GoalState]
