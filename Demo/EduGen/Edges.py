from States import MainState
from typing import List

def route_content_generation(state: MainState) -> List[str]:
    """
    사용자의 요청에 따라 다음에 실행할 콘텐츠 생성 노드를 결정합니다.
    - 리스트를 반환하여 여러 노드를 병렬로 실행시킬 수 있습니다.
    """
    print("\n--- 라우팅: 생성할 콘텐츠 결정 ---")
    content_requests = state["requirements"].content_requests
    
    next_nodes = []
    if "학습 목표 생성" in content_requests:
        next_nodes.append("generate_learning_goals")
    if "문제 생성" in content_requests:
        next_nodes.append("generate_problems")
    
    print(f"-> 다음 노드(병렬 실행): {next_nodes}")
    return next_nodes

def route_retrieve_metadata(state: MainState):
    """
    학년 메타데이터 검증의 결과값을 바탕으로 다음 노드로 진행할건지 다시 검색할지 결정합니다
    """
    binary_score = state["binary_score"]
    
    if binary_score == "yes":
        next_node = route_content_generation(state)
    else:
        next_node = "retrieve_from_vectordb"
    return next_node