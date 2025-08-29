from langgraph.graph import StateGraph, END
from States import MainState

from Nodes import extract_requirements_node
from Nodes import retrieve_from_db_node
from Nodes import generate_learning_goals_node
from Nodes import generate_problems_node
from Nodes import consolidate_response_node
from Nodes import grade_documents
from Nodes import error_handler
from Edges import route_retrieve_metadata

def get_compiled_graph():
    # 그래프 객체 생성
    workflow = StateGraph(MainState)
    # Add Nodes
    workflow.add_node("extract_requirements", extract_requirements_node)
    workflow.add_node("retrieve_from_vectordb", retrieve_from_db_node)
    workflow.add_node("generate_learning_goals", generate_learning_goals_node)
    workflow.add_node("generate_problems", generate_problems_node)
    workflow.add_node("consolidate_response", consolidate_response_node)
    workflow.add_node("evaluation_grade", grade_documents)
    workflow.add_node("error_handler", error_handler)

    # Add Edges
    # 조건부 엣지로 들어가는 연결은 포함되면 안됨
    # 의도대로 동작하지 않음
    # ---------------------------
    # 병럴 노드 이전의 연결
    workflow.set_entry_point("extract_requirements")
    workflow.add_edge("extract_requirements", "retrieve_from_vectordb")
    workflow.add_edge("retrieve_from_vectordb", "evaluation_grade")

    # 병렬 노드 후 끝 부분 연결
    workflow.add_edge("consolidate_response", END)
    workflow.add_edge("error_handler", END)

    # Add Conditional Edges
    # 조건부 엣지: 검색 노드 이후, 라우팅 함수 결과에 따라 병렬 노드로 분기

    # 검색한 문서의 연관성을 검사한 후 다시 검색할 지 다음 노드로 진행할지 라우팅
    workflow.add_conditional_edges(
        "evaluation_grade",
        route_retrieve_metadata,
        {   "retrieve_from_vectordb":"retrieve_from_vectordb",
            "generate_learning_goals": "generate_learning_goals",
            "generate_problems": "generate_problems"
        }
    )

    # `generate_learning_goals`노드에서 에러 발생처리
    workflow.add_conditional_edges(
        "generate_learning_goals",
        lambda state: state['goal_state']['status'],
        {
            "success": "consolidate_response",  # 성공하면 결과를 종합하는 노드
            "failure": "error_handler"          # 실패하면 오류 처리 노드
        }
    )
    # `generate_problems`노드에서 에러 발생 처리
    workflow.add_conditional_edges(
        "generate_problems",
        lambda state: state['problem_state']['status'],
        {
            "success": "consolidate_response",  # 성공하면 결과를 종합하는 노드
            "failure": "error_handler"          # 실패하면 오류 처리 노드로
        }
    )
    graph = workflow.compile()
    return graph