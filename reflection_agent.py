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
import re

# ─────────────────────────────────────────────
#  Model & Tools
# ─────────────────────────────────────────────
# FIX: use TWO separate LLM instances.
#
#  worker_llm  → passed into create_react_agent; gets tools bound to it
#                internally by LangGraph. Keep this ONLY for workers.
#
#  plain_llm   → used by supervisor, critic, and final nodes.
#                Never has tools bound to it, so the model never tries
#                to emit a tool-call instead of plain text.
#
#  If you reuse a single `llm` for both, create_react_agent's internal
#  bind_tools() call "contaminates" that instance and the supervisor
#  receives tool schemas it shouldn't — causing the API to reject the
#  response with "Tool choice is none, but model called a tool".

worker_llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b'))
plain_llm  = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b'))

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

worker_add    = create_react_agent(worker_llm, tools=[add],      name="add_worker")
worker_multi  = create_react_agent(worker_llm, tools=[multiply], name="multi_worker")
worker_writer = create_react_agent(worker_llm, tools=[],         name="writer_worker")

# ─────────────────────────────────────────────
#  State  — two new fields added for Reflection
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages:        Annotated[list, operator.add]
    next:            str
    reasoning:       str
    worker_result:   str
    iterations:      int
    # ── NEW ──────────────────────────────────
    critique:        str   # critic's last written critique
    quality_score:   int   # critic's numeric score (1-10)
    reflection_count: int  # how many reflection loops have occurred

# ─────────────────────────────────────────────
#  Supervisor  (unchanged logic)
# ─────────────────────────────────────────────
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

    # ── Include the critique in history so the supervisor is aware ──
    if state.get("critique"):
        history_lines.append(f"Critique (score {state.get('quality_score',0)}/10): {state['critique']}")

    prompt = supervisor_prompt.format(history="\n".join(history_lines))
    response = plain_llm.invoke([SystemMessage(content=prompt)])
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
#  Workers  (unchanged)
# ─────────────────────────────────────────────
def add_node(state: AgentState) -> AgentState:
    print("[Worker] add_worker executing...")
    result = worker_add.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    print(f"[Worker] Result: {answer}")
    return {"messages": [AIMessage(content=f"[add_worker]: {answer}")], "worker_result": answer}

def multi_node(state: AgentState) -> AgentState:
    print("[Worker] multi_worker executing...")
    result = worker_multi.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    print(f"[Worker] Result: {answer}")
    return {"messages": [AIMessage(content=f"[multi_worker]: {answer}")], "worker_result": answer}

def writer_node(state: AgentState) -> AgentState:
    print("[Worker] writer_worker executing...")
    result = worker_writer.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    print(f"[Worker] Result: {answer}")
    return {"messages": [AIMessage(content=f"[writer_worker]: {answer}")], "worker_result": answer}

# ─────────────────────────────────────────────
#  NEW: Critic Node  (the heart of Reflection)
# ─────────────────────────────────────────────
QUALITY_THRESHOLD = 7   # scores below this trigger a redo
MAX_REFLECTIONS   = 3   # hard cap on reflection loops

critic_prompt = """You are a strict but fair quality critic. A worker agent just produced an answer.
Your job is to evaluate it and provide structured feedback.

Original user question:
{question}

Worker's answer:
{answer}

Respond in EXACTLY this format (no extra text):
Score: <integer 1-10>
Critique: <one or two sentences explaining the score. If the answer is correct and complete, say so.>

Scoring guide:
- 9-10: Perfect. Correct, complete, well-written.
-  7-8: Good. Minor issues only.
-  5-6: Acceptable but missing detail or slightly off.
-  3-4: Significant errors or missing key information.
-  1-2: Wrong or completely unhelpful.
"""

def critic_node(state: AgentState) -> AgentState:
    """
    Evaluates the latest worker answer.
    Writes `critique` and `quality_score` into state.
    Does NOT change `next`; routing is handled by `after_critic`.
    """
    # ── Find the original user question ──
    question = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "Unknown question"
    )

    # ── Find the most recent worker answer ──
    answer = state.get("worker_result", "")
    if not answer:
        # fallback: last AIMessage content
        ai_msgs = [m.content for m in state["messages"] if isinstance(m, AIMessage)]
        answer = ai_msgs[-1] if ai_msgs else "No answer produced."

    prompt = critic_prompt.format(question=question, answer=answer)
    response = plain_llm.invoke([SystemMessage(content=prompt)])
    raw = response.content.strip()

    # ── Parse Score and Critique ──
    score = 0
    critique_text = raw
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
#  Routing helpers
# ─────────────────────────────────────────────
def should_continue(state: AgentState) -> Literal["critic_node", "final_node"]:
    """
    After every worker: always send to the critic first.
    (The critic then decides whether to loop or finish.)
    """
    if state.get("iterations", 0) >= 5:
        print("[should_continue] Max iterations → STOP")
        return "final_node"
    if state.get("next") == "finish":
        return "final_node"
    print(f"[should_continue] Sending to critic...")
    return "critic_node"

def after_critic(state: AgentState) -> Literal["supervisor", "final_node"]:
    """
    After the critic scores the answer:
    - If score is HIGH enough  → final_node (we're done)
    - If score is too LOW      → supervisor (try again)
    - If reflection cap hit    → final_node (safety stop)
    """
    score      = state.get("quality_score", 0)
    ref_count  = state.get("reflection_count", 0)

    if ref_count >= MAX_REFLECTIONS:
        print(f"[after_critic] Max reflections ({MAX_REFLECTIONS}) reached → STOP")
        return "final_node"

    if score >= QUALITY_THRESHOLD:
        print(f"[after_critic] Score {score} ≥ {QUALITY_THRESHOLD} → STOP (good enough)")
        return "final_node"

    print(f"[after_critic] Score {score} < {QUALITY_THRESHOLD} → LOOP (reflection {ref_count})")
    return "supervisor"

def route_supervisor(state: AgentState) -> Literal["add_node", "multi_node", "writer_node", "final_node"]:
    mapping = {
        "add_worker":    "add_node",
        "multi_worker":  "multi_node",
        "writer_worker": "writer_node",
        "finish":        "final_node",
    }
    return mapping.get(state["next"], "final_node")

# ─────────────────────────────────────────────
#  Final synthesis  (unchanged)
# ─────────────────────────────────────────────
def final_node(state: AgentState) -> AgentState:
    print("[Final] Synthesizing answer...")
    synthesis_prompt = (
        "Based on the conversation and worker results below, "
        "write a clean, concise final answer for the user. "
        "Do not mention workers, critics, or internal steps.\n\nHistory:\n"
        + "\n".join(m.content for m in state["messages"] if hasattr(m, "content"))
    )
    final = plain_llm.invoke([SystemMessage(content=synthesis_prompt)])
    return {"messages": [AIMessage(content=final.content)]}

# ─────────────────────────────────────────────
#  Graph assembly
# ─────────────────────────────────────────────
#
#  START → supervisor → [worker] → critic_node → (score OK?) → final_node → END
#                   ↑_______________|  (score too low, loop back)
#
graph = StateGraph(AgentState)

graph.add_node("supervisor",  supervisor_node)
graph.add_node("add_node",    add_node)
graph.add_node("multi_node",  multi_node)
graph.add_node("writer_node", writer_node)
graph.add_node("critic_node", critic_node)   # ← NEW
graph.add_node("final_node",  final_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_supervisor)

# All workers route to should_continue, which now sends them to critic first
graph.add_conditional_edges("add_node",    should_continue)
graph.add_conditional_edges("multi_node",  should_continue)
graph.add_conditional_edges("writer_node", should_continue)

# Critic decides: loop back to supervisor OR go to final
graph.add_conditional_edges("critic_node", after_critic)   # ← NEW

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
