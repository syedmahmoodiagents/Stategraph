import os
import re
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator


# llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b', provider="novita"))
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        huggingfacehub_api_token=HF_TOKEN,
        provider="novita",  # ->inference provider that will make not make request to groq anymore after implicit specification
    )
)

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# bind_tools() returns a NEW object — does NOT mutate `llm`.
# So `llm` stays clean for supervisor/critic/final nodes.
llm_with_add      = llm.bind_tools([add])
llm_with_multiply = llm.bind_tools([multiply])

TOOL_REGISTRY = {
    "add":      add,
    "multiply": multiply,
}


class AgentState(TypedDict):
    messages:          Annotated[list, operator.add]
    next:              str
    reasoning:         str
    worker_result:     str
    iterations:        int
    critique:          str
    quality_score:     int
    reflection_count:  int


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
            return response.content

        loop_messages.extend(run_tool_calls(response))

    
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

    # ── Inject critique so the supervisor knows a retry is happening ──
    if state.get("critique"):
        history_lines.append(f"Critique (score {state.get('quality_score', 0)}/10): {state['critique']}")

    prompt = supervisor_prompt.format(history="\n".join(history_lines))

    # Plain llm.invoke — no tools bound, zero risk of tool-call bleed
    response = llm.invoke([SystemMessage(content=prompt)])
    thought, action = parse_supervisor(response.content)

    if action not in {"add_worker", "multi_worker", "writer_worker", "finish"}:
        action = "writer_worker"

    print(f"\n[Supervisor] Thought : {thought}")
    print(f"[Supervisor] Action  : {action}")

    return {
        "messages":         [],
        "next":             action,
        "reasoning":        thought,
        "worker_result":    "",
        "iterations":       state.get("iterations", 0) + 1,
        "critique":         state.get("critique", ""),
        "quality_score":    state.get("quality_score", 0),
        "reflection_count": state.get("reflection_count", 0),
    }

# ─────────────────────────────────────────────
#  Worker Nodes  (pure StateGraph, no sub-agents)
# ─────────────────────────────────────────────
#
#  WHY we extract only HumanMessages:
#  On a reflection retry, state["messages"] accumulates stale AIMessages
#  from previous worker attempts and critic outputs. Passing that full
#  history to a tool-bound LLM confuses the API — it sees AIMessages with
#  no matching tool-call context and raises "Bad request".
#  Workers only need the original user question to do their job.
#
def get_user_messages(state: AgentState) -> list:
    """Extract only the original HumanMessage(s) — clean input for workers."""
    return [m for m in state["messages"] if isinstance(m, HumanMessage)]

def add_node(state: AgentState) -> AgentState:
    print("[Worker] add_worker executing...")
    answer = react_loop(llm_with_add, get_user_messages(state))
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[add_worker]: {answer}")],
        "worker_result": answer,
    }

def multi_node(state: AgentState) -> AgentState:
    print("[Worker] multi_worker executing...")
    answer = react_loop(llm_with_multiply, get_user_messages(state))
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[multi_worker]: {answer}")],
        "worker_result": answer,
    }

def writer_node(state: AgentState) -> AgentState:
    print("[Worker] writer_worker executing...")
    # No tools needed — plain llm.invoke is sufficient
    response = llm.invoke(get_user_messages(state))
    answer   = response.content
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[writer_worker]: {answer}")],
        "worker_result": answer,
    }

# ─────────────────────────────────────────────
#  Critic Node  ← heart of Reflection
# ─────────────────────────────────────────────
critic_prompt = """You are a strict but fair quality critic. A worker agent just produced an answer.

Original user question:
{question}

Worker's answer:
{answer}

Respond in EXACTLY this format (no extra text):
Score: <integer 1-10>
Critique: <one or two sentences explaining the score.>

Scoring guide:
- 9-10: Perfect. Correct, complete, well-written.
-  7-8: Good. Minor issues only.
-  5-6: Acceptable but missing detail or slightly off.
-  3-4: Significant errors or missing key information.
-  1-2: Wrong or completely unhelpful.
"""

def critic_node(state: AgentState) -> AgentState:
    question = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "Unknown question"
    )
    answer = state.get("worker_result", "") or next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        "No answer produced."
    )

    response = llm.invoke([SystemMessage(content=critic_prompt.format(question=question, answer=answer))])
    raw = response.content.strip()

    score, critique_text = 0, raw
    for line in raw.splitlines():
        if line.lower().startswith("score:"):
            try:
                score = int(re.search(r"\d+", line).group())
            except Exception:
                score = 5
        elif line.lower().startswith("critique:"):
            critique_text = line.split(":", 1)[1].strip()

    print(f"\n[Critic] Score   : {score}/10")
    print(f"[Critic] Critique: {critique_text}")

    return {
        "messages":         [AIMessage(content=f"[critic]: Score {score}/10 — {critique_text}")],
        "critique":         critique_text,
        "quality_score":    score,
        "reflection_count": state.get("reflection_count", 0) + 1,
    }

# ─────────────────────────────────────────────
#  Final Node
# ─────────────────────────────────────────────
def final_node(state: AgentState) -> AgentState:
    print("[Final] Synthesizing answer...")
    history = "\n".join(m.content for m in state["messages"] if hasattr(m, "content"))
    synthesis_prompt = (
        "Based on the conversation and worker results below, "
        "write a clean, concise final answer for the user. "
        "Do not mention workers, critics, or internal steps.\n\nHistory:\n"
        + history
    )
    final = llm.invoke([SystemMessage(content=synthesis_prompt)])
    return {"messages": [AIMessage(content=final.content)]}

# ─────────────────────────────────────────────
#  Routing
# ─────────────────────────────────────────────
def route_supervisor(state: AgentState) -> Literal["add_node", "multi_node", "writer_node", "final_node"]:
    return {
        "add_worker":    "add_node",
        "multi_worker":  "multi_node",
        "writer_worker": "writer_node",
        "finish":        "final_node",
    }.get(state["next"], "final_node")

def should_continue(state: AgentState) -> Literal["critic_node", "final_node"]:
    if state.get("iterations", 0) >= 5:
        print("[should_continue] Max iterations → STOP")
        return "final_node"
    if state.get("next") == "finish":
        return "final_node"
    return "critic_node"

def after_critic(state: AgentState) -> Literal["supervisor", "final_node"]:
    score     = state.get("quality_score", 0)
    ref_count = state.get("reflection_count", 0)

    if ref_count >= 3:
        print(f"[after_critic] Reflection cap reached → STOP")
        return "final_node"

    if score >= 7:
        print(f"[after_critic] Score {score} ≥ 7 → STOP")
        return "final_node"

    print(f"[after_critic] Score {score} < 7 → loop back (reflection {ref_count})")
    return "supervisor"

# ─────────────────────────────────────────────
#  Graph assembly
# ─────────────────────────────────────────────
#
#  START → supervisor → [worker] → critic_node
#               ↑                       |
#               |             score OK? → final_node → END
#               |                       |
#               └── score too low ──────┘
#
graph = StateGraph(AgentState)

graph.add_node("supervisor",  supervisor_node)
graph.add_node("add_node",    add_node)
graph.add_node("multi_node",  multi_node)
graph.add_node("writer_node", writer_node)
graph.add_node("critic_node", critic_node)
graph.add_node("final_node",  final_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor",  route_supervisor)
graph.add_conditional_edges("add_node",    should_continue)
graph.add_conditional_edges("multi_node",  should_continue)
graph.add_conditional_edges("writer_node", should_continue)
graph.add_conditional_edges("critic_node", after_critic)
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
            "messages":         [HumanMessage(content=query)],
            "next":             "",
            "reasoning":        "",
            "worker_result":    "",
            "iterations":       0,
            "critique":         "",
            "quality_score":    0,
            "reflection_count": 0,
        })

        print("\n Final Answer:")
        print(result["messages"][-1].content)
