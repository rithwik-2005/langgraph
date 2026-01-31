from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
import os

# =====================================================
# LOAD ENV
# =====================================================
load_dotenv()

# =====================================================
# MODEL (OpenRouter)
# =====================================================
model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.5,
    max_tokens=2048
)

# =====================================================
# STATE
# =====================================================
class PersistenceState(TypedDict, total=False):
    topic: str
    joke: str
    explanation: str

# =====================================================
# NODES
# =====================================================
def generate_joke(state: PersistenceState):
    prompt = f"Generate a funny joke on the topic: {state['topic']}"
    response = model.invoke(prompt).content
    return {"joke": response}

def generate_explanation(state: PersistenceState):
    prompt = f"Explain why this joke is funny:\n{state['joke']}"
    response = model.invoke(prompt).content
    return {"explanation": response}

# =====================================================
# GRAPH
# =====================================================
graph = StateGraph(PersistenceState)

graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)

# =====================================================
# CHECKPOINTER (PERSISTENCE)
# =====================================================
checkpointer = InMemorySaver()
workflow = graph.compile(checkpointer=checkpointer)

# =====================================================
# RUN
# =====================================================
if __name__ == "__main__":
    print("=== Joke Generator with Persistence ===")
    print("Type 'exit' to quit\n")

    thread_id = "1"

    while True:
        topic = input("Enter a topic: ")

        if topic.lower() == "exit":
            break

        result = workflow.invoke(
            {"topic": topic},
            config={"configurable": {"thread_id": thread_id}}
        )

        print("\n Joke:")
        print(result["joke"])

        print("\nExplanation:")
        print(result["explanation"])