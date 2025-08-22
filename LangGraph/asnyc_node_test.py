import asyncio
from langgraph.graph import StateGraph
from typing import TypedDict,Dict,Any
from langgraph.graph import START,END
from pprint import pprint
# Initiate State
class State(TypedDict):
    input: str
    output: str
    additional_data:str

# Initiate async node    
async def async_node(state: State) -> Dict[str,Any]:
    """
    비동기 작업을 수행하는 노드

    Args:
        state (_type_): 그래프의 전체 상태 값

    Returns:
        Dict[str,Any]: 업데이트할 필드가 포함된 딕셔너리
    """
    
    # 비동기 작업 수행
    result = await perform_async_operation(state["input"])

    # 여러 비동기 작업 동시 실행
    results = await asyncio.gather(
        fetch_data_1(),
        fetch_data_2(),
        fetch_data_3()
    )

    return {
        "output": result,
        "additional_data": results
    }

# Initiate async functions
async def perform_async_operation(data):
    """비동기 작업"""
    await asyncio.sleep(0.1)  # 시뮬레이션
    return f"Async result: {data}"

async def fetch_data_1():
    await asyncio.sleep(0.05)
    return "data_1"

async def fetch_data_2():
    await asyncio.sleep(0.05)
    return "data_2"

async def fetch_data_3():
    await asyncio.sleep(0.05)
    return "data_3"



# Define async function that calls ainvoke
async def run_async_graph(graph,config):
    # `ainvoke`를 호출하여 비동기 실행을 시작합니다.
    # 입력 값은 `GraphState` 딕셔너리 형태여야 합니다.
    result = await graph.ainvoke(
        config,
        stream_mode='values',
    )
    return result

# Execute async function in main namepspace
if __name__ == "__main__":
    init_config = State(
    {
        "input": "hello World!",
        "additional_data":"",
        "output":""
        }
    )
    # Initiate Graph add Nodes
    _graph = StateGraph(State)
    _graph.add_node('async_node',async_node)

    _graph.add_edge(START,"async_node")


    _graph.add_edge("async_node",END)


    # Compile Graph
    graph = _graph.compile()

    final_state = asyncio.run(run_async_graph(graph,init_config))
    
    print("-" * 20)
    print("최종 상태 전체:")
    pprint(final_state)
    print("-" * 20)

    print(f"async_node의 output: {final_state['output']}")
    print(f"async_node의 additional_data: {final_state['additional_data']}")