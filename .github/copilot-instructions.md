# LangGraph Workflow Examples - AI Coding Guidelines

## Project Overview
This codebase demonstrates LangGraph patterns for building stateful LLM applications. LangGraph enables graph-based workflows where nodes process state and edges control flow, supporting complex multi-step interactions with persistence.

## Architecture Patterns
- **State Management**: Use `TypedDict` for state schemas with `Annotated` fields and reducers (e.g., `add_messages`, `operator.add`) for aggregation
- **Node Functions**: Pure functions taking state dict, returning dict of updates to merge into state
- **Graph Structure**: `StateGraph` with nodes connected via edges; use conditional edges for branching logic
- **Persistence**: `MemorySaver`/`InMemorySaver` for RAM-based state across invocations with thread configs

## Key Conventions
- **State Definition**: 
  ```python
  class WorkflowState(TypedDict, total=False):
      messages: Annotated[list[BaseMessage], add_messages]
      scores: Annotated[list[int], operator.add]
  ```
- **Node Implementation**:
  ```python
  def process_node(state: WorkflowState) -> dict:
      # Extract from state
      data = state["field"]
      # Process
      result = model.invoke(data)
      # Return updates
      return {"output_field": result}
  ```
- **Graph Compilation**:
  ```python
  graph = StateGraph(WorkflowState)
  graph.add_node("node_name", process_node)
  graph.add_edge(START, "node_name")
  graph.add_conditional_edges("node_name", router_function)
  workflow = graph.compile(checkpointer=checkpointer)
  ```
- **Invocation**: `workflow.invoke(initial_state, config={"configurable": {"thread_id": "unique_id"}})`

## Workflow Types (See Examples)
- **Sequential**: Linear flow (sequential_workflow.py)
- **Parallel**: Concurrent processing with aggregation (parallel_workflow.py)
- **Iterative**: Conditional loops with max iterations (iterative_workflow.py)
- **Chatbot**: Message-based with memory (basic_chatbot.py)
- **Persistence**: State continuity across runs (persistance.py)

## Dependencies & Environment
- Install from `requriments.txt` (note: typo in filename)
- Use `python-dotenv` for API keys in `.env`
- Models: OpenRouter API for OpenAI-compatible endpoints, HuggingFace for local models
- No build process; run examples directly with `python script.py`

## Development Notes
- Focus on state schema design for complex aggregations
- Use structured outputs (`model.with_structured_output(Schema)`) for typed responses
- Thread IDs enable conversation continuity in persistent workflows
- Debug by inspecting state at each node via print statements or graph visualization