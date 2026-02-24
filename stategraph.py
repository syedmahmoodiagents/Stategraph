import os
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN=os.getenv("HF_TOKEN")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator
import re
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        huggingfacehub_api_token=HF_TOKEN,
        provider="novita",  # ->inference provider that will make not make request to groq anymore after implicit specification
    )
)

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


worker_add   = create_agent(llm, tools=[add],      name="add_worker")
worker_multi = create_agent(llm, tools=[multiply], name="multi_worker")
worker_writer = create_agent(llm, tools=[],        name="writer_worker")


class AgentState(TypedDict):
    messages: Annotated[list, operator.add]   # messages accumulate across nodes
    next: str                                  # which worker to call next


super_prompt = """You are a supervisor that routes tasks to the right worker.
Available workers:
  - add_worker    → addition tasks
  - multi_worker  → multiplication tasks
  - writer_worker → writing / general knowledge tasks

Reply with ONLY one of these exact words: add_worker, multi_worker, writer_worker"""

def supervisor_node(state: AgentState) -> AgentState:
    # Ask the LLM which worker to use
    decision = llm.invoke(
        [SystemMessage(content=super_prompt)] + state["messages"]
    )
    chosen = decision.content.strip().lower()

    # Fallback if the model hallucinates
    if chosen not in ("add_worker", "multi_worker", "writer_worker"):
        chosen = "writer_worker"

    return {"messages": [], "next": chosen}


def add_node(state: AgentState) -> AgentState:
    result = worker_add.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

def multi_node(state: AgentState) -> AgentState:
    result = worker_multi.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}

def writer_node(state: AgentState) -> AgentState:
    result = worker_writer.invoke({"messages": state["messages"]})
    return {"messages": result["messages"]}


def route(state: AgentState) -> Literal["add_node", "multi_node", "writer_node"]:
    mapping = {
        "add_worker":    "add_node",
        "multi_worker":  "multi_node",
        "writer_worker": "writer_node",
    }
    return mapping[state["next"]]


graph = StateGraph(AgentState)

graph.add_node("supervisor", supervisor_node)
graph.add_node("add_node",    add_node)
graph.add_node("multi_node",  multi_node)
graph.add_node("writer_node", writer_node)

# Entry point → supervisor
graph.add_edge(START, "supervisor")

# Supervisor decides → one of the three workers (conditional)
graph.add_conditional_edges("supervisor", route)

# Every worker → END
graph.add_edge("add_node",    END)
graph.add_edge("multi_node",  END)
graph.add_edge("writer_node", END)

app = graph.compile()

response = app.invoke({
    "messages": [HumanMessage(content="Who is the president of United States?")],
    "next": ""          
})

print("Output:\n")
print(response["messages"][-1].content)