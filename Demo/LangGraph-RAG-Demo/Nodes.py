# Define nodes
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from States import MainState,Requirements
from Helper_functions import search_metadata
from Helper_functions import get_grader
from Utils import init_llm,postgres_conn
from Settings import embedding_model_name

llm = init_llm()
db_conn = postgres_conn()

def extract_requirements_node(state: MainState) -> dict:
    """
    사용자 프롬프트를 입력받아서 메타데이터 및 요구사항을 추출
    메타데이터는 학교급, 학년, 도메인, 카테고리가 될 수 있음
    """
    prompt = state["prompt"]

    # LLM이 Pydantic 모델(Requirements)에 맞춰 구조화된 결과를 출력하도록 설정
    structured_llm = llm.with_structured_output(Requirements)
    # Pydantic 모델에 정의된 Literal 값을 가져옵니다.
    allowed_requests = Requirements.model_fields['content_requests'].annotation.__args__[0].__args__
    # 결과: ('학습 목표 생성', '문제 생성')

    system_prompt = f"""You are an expert at analyzing user requests for educational content creation.
    Extract the required information from the user's prompt.

    For the `content_requests` field, you MUST choose one or more values from the following exact list:
    {list(allowed_requests)}

    Do not use similar words or variations. For example, if the user asks for '연습 문제' or '퀴즈', you must map it to '문제 생성'. If they ask for '학습 목표'
    you must map it to '학습 목표 생성'.
    """

    # system_prompt = "You are an expert at analyzing user requests for educational content creation. Extract the required information from the user's prompt."
    user_prompt = f"다음 사용자 프롬프트에서 핵심 요구사항을 추출해줘:\n'{prompt}'"
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("user", user_prompt),
    ])
    
    chain = prompt_template | structured_llm
    extracted = chain.invoke({"prompt": prompt})
    
    print(f"-> 추출된 요구사항: {extracted}")

    return {"requirements": extracted}



def retrieve_from_db_node(state: MainState) -> dict:
    """
    추출된 요구사항 및 메타데이터 을 기반으로 벡터 DB에서 관련 메타데이터를 검색하는 노드
    """
    print("\n--- 2. 벡터 DB 검색 노드 실행 ---")
    requirements = state["requirements"]
    school_level = requirements.school_level
    grade = requirements.grade
    domain = requirements.domain
    subject = requirements.subject
    
    concated_metadata = f"{school_level} {grade} {subject} {domain}"
    model = SentenceTransformer(embedding_model_name)     
    embed_metadata = model.encode(concated_metadata)
    # gemini api로 임베딩 시도했으나 실패함
    # embed_metadata = client.models.embed_content(
    #     model="gemini-embedding-001",
    #     contents=concated_metadata,
    #     config=types.EmbedContentConfig(output_dimensionality=1024)
    # ).embeddings[0].values
    
    # 구축한 벡터 DB에서 검색
    retrieved_data = search_metadata(
        vector=embed_metadata,
        conn=db_conn
    )
    print([tuple(res.values()) for res in retrieved_data])
    
    return {
        "retrieved_docs": retrieved_data,
        "metadata":concated_metadata
    }

def consolidate_response_node(state: MainState) -> dict:
    """
    병렬로 생성된 콘텐츠들을 종합하여 최종 답변을 만듭니다.
    """
    print("\n--- 최종 답변 종합 노드 실행 ---")
    requirements = state["requirements"]
    problems = state['problem_state']['problems']
    learning_goals = state['goal_state']['learning_goals']
    

    final_response_parts = []
    final_response_parts.append(
        f"요청하신 **{requirements.school_level} {requirements.grade} {requirements.subject} - '{requirements.domain}'** 단원에 대한 콘텐츠입니다.\n"
    )

    if learning_goals:
        final_response_parts.append("### 학습 목표\n" + learning_goals.content)
    
    if problems:
        final_response_parts.append("\n### 연습 문제\n" + problems.content)
        
    final_response = "\n".join(final_response_parts)
    print("-> 최종 답변 생성 완료")
    
    return {"final_response": final_response}

def generate_learning_goals_node(state: MainState) -> dict:
    """
    입력된 정보를 바탕으로 학습 목표를 생성합니다. (병렬 처리 대상)
    """
    requirements = state["requirements"]
    docs = state["retrieved_docs"]
    try:
        prompt_template = ChatPromptTemplate.from_template(
            "{school_level} {grade} {subject} 과목의 '{domain}' 단원에 대한 학습 목표를 생성해줘.\n"
            "참고 자료:\n{docs}"
        )
    
        chain = prompt_template | llm

        learning_goals = chain.invoke({
            "school_level": requirements.school_level,
            "grade": requirements.grade,
            "subject": requirements.subject,
            "domain": requirements.domain,
            "docs": docs
        })
        return {
            "goal_state": {
                "learning_goals": learning_goals,
                "status": "success"},
        }
    except Exception as e:
        return {
            "goal_state" : {
                "error_message" : e,
                "status" : "failure"
            }
        }
        

    

def generate_problems_node(state: MainState) -> dict:
    """
    검색된 정보를 바탕으로 연습 문제를 생성합니다. (병렬 처리 대상)
    """
    requirements = state["requirements"]
    docs = state["retrieved_docs"]
    try:
        prompt_template = ChatPromptTemplate.from_template(
            "{school_level} {grade} {subject} 과목의 '{domain}' 단원에 대한 이해를 확인할 수 있는 문제를 상,중,하 수준의 문제들을 각각 1개씩 생성해줘.\n"
            "참고 자료:\n{docs}"
        )
        
        chain = prompt_template | llm
        problems = chain.invoke({
            "school_level": requirements.school_level,
            "grade": requirements.grade,
            "subject": requirements.subject,
            "domain": requirements.domain,
            "docs": docs
        })

        return {
            "problem_state": {
                "problems": problems,
                "status": "success"},
        }
    except Exception as e:
        return {
            "problem_state" : {
                "error_message" : e,
                "status" : "failure"
            }
        }
        


# 검색된 문서의 관련성 평가
def grade_documents(state:MainState):
    """
    Vector DB로부터 검색한 수업 대상의 메타데이터와 
    요구사항으로부터 추출한 메타데이터가 연관성이 있는지 검사합니다
    """
    print("==== [GRADE DOCUMENTS] ====")
    

    retrieval_grader = get_grader(llm)

    question = state["metadata"]
    documents = state["retrieved_docs"]

    # 각 문서 점수 평가
    filtered_docs = []
    for d in documents:
        joined_doc = " ".join(val for val in d.values())
        score = retrieval_grader.invoke(
            {"question": question, "document": joined_doc}
        )
        grade = score.binary_score
        if grade == "yes":
            print("==== GRADE: DOCUMENT RELEVANT ====")
            filtered_docs.append(d)
        else:
            print("==== GRADE: DOCUMENT NOT RELEVANT ====")
            continue
    # 검색된 문서의 갯수중에 70% 이상 연관 있을 때 통과
    threshold = int(round(len(documents) * 0.7))
    if len(filtered_docs) >= threshold:
        return {"binary_score": "yes"}
    else:
        return {"binary_score": "no"}

def error_handler(state: MainState) -> MainState:
    # 오류를 남기거나 사용자에게 알리거나 후속조치를 위함
    # 각 서브 상태에서 에러 메시지를 가져옴
    problem_error = state.get('problem_state', {}).get('error_message')
    goal_error = state.get('goal_state', {}).get('error_message')
    
    final_error_message = "오류가 발생했습니다.\n"
    if problem_error:
        final_error_message += f"문제 생성 오류: {problem_error}\n"
    if goal_error:
        final_error_message += f"학습 목표 생성 오류: {goal_error}\n"
        
    return {"final_response": final_error_message}