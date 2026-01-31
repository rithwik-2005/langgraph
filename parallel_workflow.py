from dotenv import load_dotenv
import os
import operator
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

# === OPENROUTER MODEL ===
model = ChatOpenAI(
    model="deepseek/deepseek-chat",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0.5,
    max_tokens=2048
)

# === STRUCTURED OUTPUT SCHEMA ===
class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed UPSC-style feedback")
    score: int = Field(description="Score between 0 and 10", ge=0, le=10)

structured_model = model.with_structured_output(EvaluationSchema)

# === STATE SCHEMA (SAFE) ===
class UPSCSchema(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# === NODES ===
def evaluate_language(state: UPSCSchema):
    prompt = f"""
You are a UPSC Civil Services examiner.

Evaluate the LANGUAGE of the following essay in terms of:
- grammar and syntax
- vocabulary
- formal tone
- coherence

Give detailed feedback and a strict score from 0 to 10.

Essay:
{state['essay']}
"""
    result = structured_model.invoke(prompt)
    return {
        "language_feedback": result.feedback,
        "individual_scores": [result.score]
    }

def evaluate_analysis(state: UPSCSchema):
    prompt = f"""
You are a UPSC Civil Services examiner.

Evaluate the DEPTH OF ANALYSIS of the following essay in terms of:
- conceptual clarity
- multidimensional coverage
- critical thinking
- balance

Give detailed feedback and a strict score from 0 to 10.

Essay:
{state['essay']}
"""
    result = structured_model.invoke(prompt)
    return {
        "analysis_feedback": result.feedback,
        "individual_scores": [result.score]
    }

def evaluate_clarity(state: UPSCSchema):
    prompt = f"""
You are a UPSC Civil Services examiner.

Evaluate the CLARITY OF THOUGHT of the following essay in terms of:
- logical flow
- paragraph transitions
- precision of expression

Give detailed feedback and a strict score from 0 to 10.

Essay:
{state['essay']}
"""
    result = structured_model.invoke(prompt)
    return {
        "clarity_feedback": result.feedback,
        "individual_scores": [result.score]
    }

def final_evaluation(state: UPSCSchema):
    summary_prompt = f"""
You are a UPSC evaluator.

Provide a holistic assessment of the essay based on:
- Language feedback: {state.get('language_feedback', '')}
- Analysis feedback: {state.get('analysis_feedback', '')}
- Clarity feedback: {state.get('clarity_feedback', '')}

Write in formal UPSC examiner tone.
"""
    summary = model.invoke(summary_prompt).content

    scores = state.get("individual_scores", [])
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "overall_feedback": summary,
        "avg_score": avg_score
    }

# === GRAPH ===
graph = StateGraph(UPSCSchema)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_clarity", evaluate_clarity)
graph.add_node("final_evaluation", final_evaluation)

# fan-out
graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_clarity")

# fan-in
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_clarity", "final_evaluation")

graph.add_edge("final_evaluation", END)

workflow = graph.compile()

# === RUN ===
if __name__ == "__main__":
    essay = input("Write UPSC essay:\n\n")

    result = workflow.invoke({
        "essay": essay,
        "individual_scores": []   # CRITICAL FIX
    })

    print("\n=== UPSC EVALUATION ===\n")
    print(result["overall_feedback"])
    print(f"\nAverage Score: {result['avg_score']:.2f}/10")

