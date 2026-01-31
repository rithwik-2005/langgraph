from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import TypedDict, Literal
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import os

#ENV
load_dotenv()
# OPENROUTER MODELS
generator_llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.7
)

evaluator_llm = ChatOpenAI(
    model="qwen/qwen-2.5-72b-instruct",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.0
)

optimizer_llm = ChatOpenAI(
    model="deepseek/deepseek-chat",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    temperature=0.3
)

# STRUCTURED OUTPUT
class TweetEvaluation(BaseModel):
    evaluation: Literal["approved", "needs_improvement"] = Field(
        description="Final decision on tweet quality"
    )
    feedback: str = Field(description="Concise critique of the tweet")

structured_evaluator_llm = evaluator_llm.with_structured_output(TweetEvaluation)

# STATE
class TweetState(TypedDict, total=False):
    topic: str
    tweet: str
    evaluation: Literal["approved", "needs_improvement"]
    feedback: str
    iteration: int
    max_iteration: int

# NODES
def generate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a funny and clever Twitter/X influencer."),
        HumanMessage(content=f"""
Write a short, original, and hilarious tweet on the topic: "{state['topic']}".

Rules:
- Max 280 characters
- NOT Q&A style
- Observational humor, sarcasm, or irony
- Simple everyday English
- Output ONLY the tweet text
""")
    ]
    tweet = generator_llm.invoke(messages).content.strip()
    return {"tweet": tweet}

def evaluate_tweet(state: TweetState):
    messages = [
        SystemMessage(content="You are a ruthless Twitter/X critic."),
        HumanMessage(content=f"""
Evaluate the tweet below.

Tweet:
{state['tweet']}

Reject if:
- More than 280 characters
- Q&A format
- Contains explanations, headings, or meta commentary

Respond ONLY in structured format.
""")
    ]
    result = structured_evaluator_llm.invoke(messages)
    return {
        "evaluation": result.evaluation,
        "feedback": result.feedback
    }

def optimize_tweet(state: TweetState):
    messages = [
        SystemMessage(content="""
You are a professional Twitter/X ghostwriter.

STRICT RULES:
- Output ONLY the tweet text
- NO explanations
- NO markdown
- NO bullet points
- NO headings
- Under 280 characters
"""),
        HumanMessage(content=f"""
Feedback:
{state['feedback']}

Topic:
{state['topic']}

Rewrite the tweet to be funnier and more viral.
""")
    ]
    tweet = optimizer_llm.invoke(messages).content.strip()
    tweet = tweet.split("\n")[0]  # safety guard

    return {
        "tweet": tweet,
        "iteration": state.get("iteration", 0) + 1
    }

# ROUTING
def route_evaluation(state: TweetState):
    if state.get("evaluation") == "approved":
        return "approved"

    if state.get("iteration", 0) >= state.get("max_iteration", 0):
        return "approved"

    return "needs_improvement"

# GRAPH
graph = StateGraph(TweetState)

graph.add_node("generate", generate_tweet)
graph.add_node("evaluate", evaluate_tweet)
graph.add_node("optimize", optimize_tweet)

graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")

graph.add_conditional_edges(
    "evaluate",
    route_evaluation,
    {
        "approved": END,
        "needs_improvement": "optimize"
    }
)

graph.add_edge("optimize", "evaluate")

workflow = graph.compile()

# RUN
if __name__ == "__main__":
    topic = input("Give topic: ")

    result = workflow.invoke({
        "topic": topic,
        "iteration": 1,
        "max_iteration": 5
    })

    print("\n==== Final Tweet ====\n")
    print(result["tweet"])
