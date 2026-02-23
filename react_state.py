import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator


llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b'))


def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


worker_add    = create_react_agent(llm, tools=[add],      name="add_worker")
worker_multi  = create_react_agent(llm, tools=[multiply], name="multi_worker")
worker_writer = create_react_agent(llm, tools=[],         name="writer_worker")


class AgentState(TypedDict):
    messages:      Annotated[list, operator.add]  # full conversation history
    next:          str                            # worker to call
    reasoning:     str                            # supervisor's reasoning trace
    worker_result: str                            # last worker's answer
    iterations:    int                            # loop counter (safety cap)

MAX_ITERATIONS = 5

SUPERVISOR_PROMPT = """You are a supervisor that follows the ReAct pattern: Reason then Act.

At every step you MUST output in exactly this format (no extra text):

Thought: <explain what the user wants and which worker is best suited>
Action: <one of: add_worker | multi_worker | writer_worker | FINISH>

Rules:
- Use add_worker    for addition tasks
- Use multi_worker  for multiplication tasks
- Use writer_worker for writing, general knowledge, or questions
- Use FINISH        only when you already have a complete answer in the conversation

Conversation so far:
{history}
"""


def parse_supervisor(text: str):
    """Extract Thought and Action from the supervisor's ReAct output."""
    thought, action = "", ""
    for line in text.strip().splitlines():
        if line.lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        elif line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip().lower()
    return thought, action

VALID_ACTIONS = {"add_worker", "multi_worker", "writer_worker", "finish"}



def supervisor_node(state: AgentState) -> AgentState:
    """
    REASON step — the supervisor reads the full history,
    writes a Thought, and picks an Action.
    """
    # Build a readable history string for the prompt
    history_lines = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            history_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            history_lines.append(f"Assistant: {m.content}")
        elif isinstance(m, ToolMessage):
            history_lines.append(f"Tool Result: {m.content}")

    # Append the last worker result so the supervisor can observe it
    if state.get("worker_result"):
        history_lines.append(f"Worker Observation: {state['worker_result']}")

    prompt = SUPERVISOR_PROMPT.format(history="\n".join(history_lines))
    response = llm.invoke([SystemMessage(content=prompt)])

    thought, action = parse_supervisor(response.content)

    # Fallback if parsing fails
    if action not in VALID_ACTIONS:
        action = "writer_worker"

    print(f"\n[Supervisor] Thought : {thought}")
    print(f"[Supervisor] Action  : {action}")

    return {
        "messages":      [],               # no new chat messages yet
        "next":          action,
        "reasoning":     thought,
        "worker_result": "",               # reset for next iteration
        "iterations":    state.get("iterations", 0) + 1,
    }


def add_node(state: AgentState) -> AgentState:
    """ACTION step — delegate to the add worker."""
    print("[Worker] add_worker executing...")
    result = worker_add.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[add_worker]: {answer}")],
        "worker_result": answer,
    }


def multi_node(state: AgentState) -> AgentState:
    """ACTION step — delegate to the multiply worker."""
    print("[Worker] multi_worker executing...")
    result = worker_multi.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[multi_worker]: {answer}")],
        "worker_result": answer,
    }


def writer_node(state: AgentState) -> AgentState:
    """ACTION step — delegate to the writer worker."""
    print("[Worker] writer_worker executing...")
    result = worker_writer.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[writer_worker]: {answer}")],
        "worker_result": answer,
    }


def final_node(state: AgentState) -> AgentState:
    """
    Synthesize a clean final answer from the full conversation history.
    """
    print("[Final] Synthesizing answer...")

    synthesis_prompt = """Based on the conversation and worker results below, 
write a clean, concise final answer for the user. Do not mention workers or internal steps.

History:
""" + "\n".join(
        m.content for m in state["messages"] if hasattr(m, "content")
    )

    final = llm.invoke([SystemMessage(content=synthesis_prompt)])
    return {
        "messages": [AIMessage(content=final.content)],
    }


# ─── Routing Functions ───────────────────────────────────────────────────────

def route_supervisor(state: AgentState) -> Literal[
    "add_node", "multi_node", "writer_node", "final_node"
]:
    """After supervisor reasons, pick which node to run next."""
    # Hard stop if we've looped too many times
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        print("[Router] Max iterations reached — forcing FINISH")
        return "final_node"

    mapping = {
        "add_worker":    "add_node",
        "multi_worker":  "multi_node",
        "writer_worker": "writer_node",
        "finish":        "final_node",
    }
    return mapping.get(state["next"], "final_node")


def route_after_worker(state: AgentState) -> Literal["supervisor", END]:
    """
    After a worker runs, loop back to the supervisor so it can
    OBSERVE the result and decide whether to act again or FINISH.
    """
    if state.get("iterations", 0) >= MAX_ITERATIONS:
        return END
    return "supervisor"   # ← this is the ReAct loop-back


# ─── Build the Graph ─────────────────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("supervisor",  supervisor_node)
graph.add_node("add_node",    add_node)
graph.add_node("multi_node",  multi_node)
graph.add_node("writer_node", writer_node)
graph.add_node("final_node",  final_node)

# Entry
graph.add_edge(START, "supervisor")

# Supervisor → conditional worker or finish
graph.add_conditional_edges("supervisor", route_supervisor)

# Workers → loop back to supervisor (ReAct observe step)
graph.add_conditional_edges("add_node",    route_after_worker)
graph.add_conditional_edges("multi_node",  route_after_worker)
graph.add_conditional_edges("writer_node", route_after_worker)

# Final answer → done
graph.add_edge("final_node", END)

app = graph.compile()



queries = [
    "What is 12 + 45?",
    "Multiply 6 by 9, then write a short poem about the result.",
    "Who is the president of the United States?",
]

for query in queries:
    print("\n" + "="*60)
    print(f"Query: {query}")
    print("="*60)

    result = app.invoke({
        "messages":      [HumanMessage(content=query)],
        "next":          "",
        "reasoning":     "",
        "worker_result": "",
        "iterations":    0,
    })

    print("\n Final Answer:")
    print(result["messages"][-1].content)


