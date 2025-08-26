# LangChain & LangGraph Study Repository

This repository contains a collection of Jupyter notebooks and Python scripts for studying and practicing concepts related to LangChain and LangGraph.

## LangChain Examples

This section provides an overview of the LangChain examples included in this repository.

*   **[test_agent.ipynb](./LangChain/test_agent.ipynb)**: Covers the basics of creating and running LangChain agents.
*   **[test_chatbot.ipynb](./LangChain/test_chatbot.ipynb)**: Demonstrates how to build a simple chatbot using LangChain.
*   **[test_memory.ipynb](./LangChain/test_memory.ipynb)**: Explores different memory types and how to use them in conversational applications.
*   **[test_runnable.ipynb](./LangChain/test_runnable.ipynb)**: Introduces the LangChain Runnable protocol for creating custom chains.

## LangGraph Examples

This section provides an overview of the LangGraph examples included in this repository.

### Core Concepts

*   **[Basic.ipynb](./LangGraph/Basic.ipynb)**: Introduces the fundamental components of LangGraph, including `State`, `Node`, and `Edge`.
*   **[StateGraph.ipynb](./LangGraph/StateGraph.ipynb)**: Focuses on state management within graphs, a core concept in LangGraph.
*   **[Node_Concept.ipynb](./LangGraph/Node_Concept.ipynb)**: Provides an in-depth look at Nodes, their features, and design principles.
*   **[Node_Patterns.ipynb](./LangGraph/Node_Patterns.ipynb)**: Explores various node implementation patterns, from synchronous and asynchronous nodes to conditional and class-based nodes.

### Advanced Features

*   **[CallingTool.ipynb](./LangGraph/CallingTool.ipynb)**: Demonstrates how to define and use tools within a LangGraph agent.
*   **[ParrllelNode.ipynb](./LangGraph/ParrllelNode.ipynb)**: Shows how to execute nodes in parallel for improved performance.
*   **[SubGraph.ipynb](./LangGraph/SubGraph.ipynb)**: Covers the creation and management of subgraphs to build modular and complex systems.
*   **[Streaming_Stdout.ipynb](./LangGraph/Streaming_Stdout.ipynb)**: Illustrates how to stream output from a LangGraph execution to stdout.

### Agentic Patterns

*   **[HistoryManage.ipynb](./LangGraph/HistoryManage.ipynb)**: Provides examples of techniques for managing conversation history, including manual and automatic message removal, and creating conversation summaries.
*   **[HITL.ipynb](./LangGraph/HITL.ipynb)**: Explains and demonstrates how to implement Human-in-the-Loop (HITL) workflows for tasks that require human approval or intervention.

### Supporting Files

*   **[asnyc_node_test.py](./LangGraph/asnyc_node_test.py)**: A Python script for testing the execution of asynchronous nodes.
*   **[configs.py](./LangGraph/configs.py)**: A configuration file for storing shared settings, such as API keys and LLM definitions.

## How to Run

1.  **Install Dependencies**
    * This project uses `uv` for dependency management.
    * To install the dependencies, run below command
    ```bash
    uv sync
    ```

2.  **Set up Environment Variables**
    * Create a `.env` file in the root directory and add your API keys, following the structure in `configs.py`.
    * After filling in the values for the keys in the `dummy.env` file, rename it to `.env`.