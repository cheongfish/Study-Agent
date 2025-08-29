from langchain_core.prompts import ChatPromptTemplate
from States import GradeDocuments

def search_metadata(
    vector,
    conn,
    k=3,
    ):
    """pgvector에서 관련성 높은 문서를 k개 검색합니다."""
    with conn.cursor() as cursor:
        # 벡터 검색 시 필요한 모든 컬럼을 가져옵니다.
        cursor.execute(
            """SELECT basecode, content,school_level, grade, domain, category
                FROM curriculum
                ORDER BY embedding <=> %s::vector LIMIT %s""",
            (list(vector), k)
        )
        # 결과를 딕셔너리 리스트로 변환
        results = [
            dict(zip([desc[0] for desc in cursor.description], row))
            for row in cursor.fetchall()
        ]
        return results
def get_grader(llm):
    # GradeDocuments 데이터 모델을 사용하여 LLM의 구조화된 출력 생성
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    # 시스템 프롬프트 정의: 검색된 문서가 사용자 질문에 관련이 있는지 평가하는 시스템 역할 정의
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""

    # 채팅 프롬프트 템플릿 생성
    grade_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    # 검색 평가기 생성
    retrieval_grader = grade_prompt | structured_llm_grader
    return retrieval_grader