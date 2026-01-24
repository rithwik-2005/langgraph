#building Q&A bot
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint

load_dotenv()         #.env is loaded

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    temperature=0.5
)
model = ChatHuggingFace(llm=llm)     #model is defined
#model=ChatOpenAI(model="gpt-4.1-mini")
#create the state
class llmstate(TypedDict):
    question:str
    answer:str

#define the function of a node
def llm_qa(state:llmstate):
    
    #extract the question from state
    question=state["question"]
    #form a prompt
    prompt=f'answer the following question:\n {question}'
    #ask that question to the llm
    answer=model.invoke(prompt).content
    #update the state
    return {'answer':answer}

#create the graph
graph=StateGraph(llmstate)

#add nodes
graph.add_node("llm_qa",llm_qa)
#add edges
graph.add_edge(START,"llm_qa")
graph.add_edge("llm_qa",END)

#complie the graph
workflow=graph.compile()

#-------run example-------
if __name__=="__main__":
    ask=input("enter the question:")
    initial_state={
        "question":ask
    }
    final_state=workflow.invoke(initial_state)
    print(final_state["answer"])
