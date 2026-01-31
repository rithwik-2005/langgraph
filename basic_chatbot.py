from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os


"""BaseMessage is the parent class for all conversation messages
in LangChain,defining what a "message" is at a fundamental level.
It defines the common structure used by HumanMessage,AIMessage and other Message
"""

load_dotenv()
# STATE
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# MODEL (OpenRouter)

model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.5,
    max_tokens=2048
)

# NODE

def chat_node(state: ChatState):
    messages = state["messages"]
    response = model.invoke(messages)
    return {"messages": [response]}

# GRAPH

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

workflow = graph.compile()

# RUN

if __name__ == "__main__":

    while True:

        human_message = input("Please ask the question to the chat model:\n")
        if human_message.strip().lower() in ["exit","quit","bye"]:
            break
        result = workflow.invoke(
        {
            "messages": [HumanMessage(content=human_message)]
        }
        )
        print("\n=== AI RESPONSE ===\n")
        print(result["messages"][-1].content)
    print("THANK-YOU")

    
