import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator

llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', provider="novita"))


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

llm_with_add      = llm.bind_tools([add])
llm_with_multiply = llm.bind_tools([multiply])

# Tool registry: name → callable
TOOL_REGISTRY = {
    "add":      add,
    "multiply": multiply,
}


class AgentState(TypedDict):
    messages:      Annotated[list, operator.add]
    next:          str
    reasoning:     str
    worker_result: str
    iterations:    int


def run_tool_calls(ai_message: AIMessage) -> list[ToolMessage]:
    """Execute every tool call in an AIMessage, return ToolMessages."""
    results = []
    for tc in ai_message.tool_calls:
        fn = TOOL_REGISTRY.get(tc["name"])
        try:
            output = fn.invoke(tc["args"]) if fn else f"Unknown tool: {tc['name']}"
        except Exception as e:
            output = f"Tool error: {e}"
        results.append(ToolMessage(
            tool_call_id=tc["id"],
            name=tc["name"],
            content=str(output),
        ))
    return results

def react_loop(bound_llm, messages: list, max_steps: int = 5) -> str:
    """
    Manual ReAct loop:
      1. Call the tool-bound LLM
      2. If it emits tool_calls → execute → feed results back → repeat
      3. If it emits plain text  → return it as the final answer
    """
    loop_messages = list(messages)

    for _ in range(max_steps):
        response = bound_llm.invoke(loop_messages)
        loop_messages.append(response)

        if not getattr(response, "tool_calls", []):
            return response.content   # plain text = final answer

        tool_results = run_tool_calls(response)
        loop_messages.extend(tool_results)

    # Safety fallback
    for m in reversed(loop_messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", []):
            return m.content
    return "Max steps reached without a final answer."

supervisor_prompt = """You are a supervisor that follows the ReAct pattern: Reason then Act.

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
    thought, action = "", ""
    for line in text.strip().splitlines():
        if line.lower().startswith("thought:"):
            thought = line.split(":", 1)[1].strip()
        elif line.lower().startswith("action:"):
            action = line.split(":", 1)[1].strip().lower()
    return thought, action

def supervisor_node(state: AgentState) -> AgentState:
    history_lines = []
    for m in state["messages"]:
        if isinstance(m, HumanMessage):
            history_lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage):
            history_lines.append(f"Assistant: {m.content}")
        elif isinstance(m, ToolMessage):
            history_lines.append(f"Tool Result: {m.content}")

    if state.get("worker_result"):
        history_lines.append(f"Worker Observation: {state['worker_result']}")

    prompt = supervisor_prompt.format(history="\n".join(history_lines))

    # Plain llm.invoke — no tools bound, zero risk of tool-call bleed
    response = llm.invoke([SystemMessage(content=prompt)])
    thought, action = parse_supervisor(response.content)

    if action not in {"add_worker", "multi_worker", "writer_worker", "finish"}:
        action = "writer_worker"

    print(f"\n[Supervisor] Thought : {thought}")
    print(f"[Supervisor] Action  : {action}")

    return {
        "messages":      [],
        "next":          action,
        "reasoning":     thought,
        "worker_result": "",
        "iterations":    state.get("iterations", 0) + 1,
    }


def add_node(state: AgentState) -> AgentState:
    print("[Worker] add_worker executing...")
    answer = react_loop(llm_with_add, state["messages"])
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[add_worker]: {answer}")],
        "worker_result": answer,
    }

def multi_node(state: AgentState) -> AgentState:
    print("[Worker] multi_worker executing...")
    answer = react_loop(llm_with_multiply, state["messages"])
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[multi_worker]: {answer}")],
        "worker_result": answer,
    }

def writer_node(state: AgentState) -> AgentState:
    print("[Worker] writer_worker executing...")
    # No tools needed — plain llm.invoke is sufficient
    response = llm.invoke(state["messages"])
    answer   = response.content
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[writer_worker]: {answer}")],
        "worker_result": answer,
    }

def final_node(state: AgentState) -> AgentState:
    print("[Final] Synthesizing answer...")
    history = "\n".join(m.content for m in state["messages"] if hasattr(m, "content"))
    synthesis_prompt = (
        "Based on the conversation and worker results below, "
        "write a clean, concise final answer for the user. "
        "Do not mention workers or internal steps.\n\nHistory:\n"
        + history
    )
    final = llm.invoke([SystemMessage(content=synthesis_prompt)])
    return {"messages": [AIMessage(content=final.content)]}

def route_supervisor(state: AgentState) -> Literal["add_node", "multi_node", "writer_node", "final_node"]:
    return {
        "add_worker":    "add_node",
        "multi_worker":  "multi_node",
        "writer_worker": "writer_node",
        "finish":        "final_node",
    }.get(state["next"], "final_node")

def should_continue(state: AgentState) -> Literal["supervisor", "final_node"]:
    if state.get("iterations", 0) >= 5:
        print(f"[should_continue] Max iterations reached → STOP")
        return "final_node"
    if state.get("next") == "finish":
        print("[should_continue] Supervisor said FINISH → STOP")
        return "final_node"
    print(f"[should_continue] Iteration {state['iterations']} — CONTINUE")
    return "supervisor"

graph = StateGraph(AgentState)

graph.add_node("supervisor",  supervisor_node)
graph.add_node("add_node",    add_node)
graph.add_node("multi_node",  multi_node)
graph.add_node("writer_node", writer_node)
graph.add_node("final_node",  final_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_supervisor)
graph.add_conditional_edges("add_node",    should_continue)
graph.add_conditional_edges("multi_node",  should_continue)
graph.add_conditional_edges("writer_node", should_continue)
graph.add_edge("final_node", END)

app = graph.compile()

# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    queries = [
        "What is 12 + 45?",
        "Multiply 6 by 9, then write a short poem about the result.",
        "Who is the president of the United States?",
    ]

    for query in queries:
        print("\n" + "=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        result = app.invoke({
            "messages":      [HumanMessage(content=query)],
            "next":          "",
            "reasoning":     "",
            "worker_result": "",
            "iterations":    0,
        })

        print("\n Final Answer:")
        print(result["messages"][-1].content)