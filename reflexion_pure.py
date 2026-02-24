import os
import json
from dotenv import load_dotenv
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, Literal
import operator
import re

# ─────────────────────────────────────────────
#  ONE shared LLM — no create_react_agent at all
# ─────────────────────────────────────────────
# llm = ChatHuggingFace(llm=HuggingFaceEndpoint(repo_id='openai/gpt-oss-120b'))
llm = ChatHuggingFace(
    llm=HuggingFaceEndpoint(
        repo_id="openai/gpt-oss-120b",
        huggingfacehub_api_token=HF_TOKEN,
        provider="novita",  # ->inference provider that will make not make request to groq anymore after implicit specification
    )
)

# ─────────────────────────────────────────────
#  Tool definitions  (@tool decorator for schema)
# ─────────────────────────────────────────────
@tool
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

# Pre-bind LLM variants for each worker's specific tool set.
# bind_tools() returns a NEW object each time — it does NOT mutate `llm`.
# So `llm` itself stays clean; only the bound variants carry tool schemas.
llm_with_add      = llm.bind_tools([add])
llm_with_multiply = llm.bind_tools([multiply])

# Tool registry: name → callable python function
TOOL_REGISTRY = {
    "add":      add,
    "multiply": multiply,
}

# ─────────────────────────────────────────────
#  State
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    messages:          Annotated[list, operator.add]
    next:              str
    reasoning:         str
    worker_result:     str
    iterations:        int
    critique:          str
    quality_score:     int
    reflection_count:  int
    reflexion_memory:  Annotated[list, operator.add]

# ─────────────────────────────────────────────
#  Shared tool-execution helper
# ─────────────────────────────────────────────
def run_tool_calls(ai_message: AIMessage) -> list[ToolMessage]:
    """
    Given an AIMessage that may contain tool_calls,
    execute each tool and return a list of ToolMessages.
    """
    results = []
    for tc in ai_message.tool_calls:
        fn   = TOOL_REGISTRY.get(tc["name"])
        args = tc["args"]
        try:
            output = fn.invoke(args) if fn else f"Unknown tool: {tc['name']}"
        except Exception as e:
            output = f"Tool error: {e}"
        results.append(ToolMessage(
            tool_call_id=tc["id"],
            name=tc["name"],
            content=str(output),
        ))
    return results

def extract_final_answer(messages: list) -> str:
    """
    Walk the message list in reverse and return the first
    plain-text AIMessage content (i.e., no tool calls pending).
    """
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not getattr(m, "tool_calls", []):
            return m.content
    return ""

# ─────────────────────────────────────────────
#  Manual ReAct loop used by tool-capable workers
#
#  Implements the pattern that create_react_agent used to do:
#    1. Call LLM (with tools bound)
#    2. If model emits tool_calls → execute them → loop
#    3. If model emits plain text → done
# ─────────────────────────────────────────────
def react_loop(bound_llm, user_messages: list, max_steps: int = 5) -> tuple[str, list]:
    """
    Run a ReAct loop with the given tool-bound LLM.
    Returns (final_answer_str, all_messages_including_tool_results).
    """
    loop_messages = list(user_messages)   # local copy; don't mutate state

    for _ in range(max_steps):
        response = bound_llm.invoke(loop_messages)
        loop_messages.append(response)

        if not getattr(response, "tool_calls", []):
            # No tool calls → model produced a final answer
            return response.content, loop_messages

        # Execute every tool the model requested
        tool_results = run_tool_calls(response)
        loop_messages.extend(tool_results)

    # Safety: extract whatever text is available
    return extract_final_answer(loop_messages) or "Max steps reached without answer.", loop_messages

# ─────────────────────────────────────────────
#  Worker Nodes  (pure StateGraph, no sub-agents)
# ─────────────────────────────────────────────
#
#  WHY we extract only HumanMessages:
#  On a reflexion retry, state["messages"] accumulates stale AIMessages from
#  previous worker attempts, critic scores, and reflexion lessons.
#  Passing that full history into a tool-bound LLM causes "Bad request" —
#  the API sees AIMessages with no matching tool-call context.
#  Workers only ever need the original user question.
#
def get_user_messages(state: AgentState) -> list:
    """Extract only the original HumanMessage(s) — clean input for workers."""
    return [m for m in state["messages"] if isinstance(m, HumanMessage)]

def add_node(state: AgentState) -> AgentState:
    print("[Worker] add_worker executing...")
    answer, _ = react_loop(llm_with_add, get_user_messages(state))
    print(f"[Worker] Result: {answer}")
    return {
        "messages":      [AIMessage(content=f"[add_worker]: {answer}")],
        "worker_result": answer,
    }

def multi_node(state: AgentState) -> AgentState:
    print("[Worker] multi_worker executing...")
    answer, _ = react_loop(llm_with_multiply, get_user_messages(state))
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
#  Supervisor
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

{memory_section}

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
    if state.get("critique"):
        history_lines.append(f"Critique (score {state.get('quality_score', 0)}/10): {state['critique']}")

    memory = state.get("reflexion_memory", [])
    if memory:
        mem_lines = "\n".join(f"  {i+1}. {m}" for i, m in enumerate(memory))
        memory_section = f"Lessons learned from previous failed attempts:\n{mem_lines}\n"
    else:
        memory_section = ""

    prompt = supervisor_prompt.format(
        history="\n".join(history_lines),
        memory_section=memory_section,
    )
    # Plain llm.invoke — no tools bound here, no risk of tool-call bleed
    response = llm.invoke([SystemMessage(content=prompt)])
    thought, action = parse_supervisor(response.content)

    if action not in {"add_worker", "multi_worker", "writer_worker", "finish"}:
        action = "writer_worker"

    print(f"\n[Supervisor] Thought : {thought}")
    print(f"[Supervisor] Action  : {action}")
    if memory:
        print(f"[Supervisor] Memory  : {len(memory)} lesson(s) loaded")

    return {
        "messages":         [],
        "next":             action,
        "reasoning":        thought,
        "worker_result":    "",
        "iterations":       state.get("iterations", 0) + 1,
        "critique":         state.get("critique", ""),
        "quality_score":    state.get("quality_score", 0),
        "reflection_count": state.get("reflection_count", 0),
        "reflexion_memory": [],
    }

# ─────────────────────────────────────────────
#  Critic Node
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
    answer = state.get("worker_result", "") or (
        next((m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)), "No answer.")
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
        "reflexion_memory": [],
    }

# ─────────────────────────────────────────────
#  Reflexion Node
# ─────────────────────────────────────────────
reflexion_prompt = """You are a self-improvement coach for an AI agent.

The agent just failed to answer a question well. Your job is to write a short,
specific, actionable lesson that will help it do BETTER on the next attempt.

User question:
{question}

Failed answer:
{answer}

Critic's feedback (score {score}/10):
{critique}

Previous lessons (so you don't repeat them):
{prior_lessons}

Write ONE concise lesson (1-2 sentences) starting with an action verb.
Example good lessons:
- "Double-check arithmetic results before reporting the final number."
- "When asked for a poem, always include at least four lines with a consistent rhythm."
- "State the source or basis for factual claims about current events."

Your lesson:"""

def reflexion_node(state: AgentState) -> AgentState:
    question = next(
        (m.content for m in state["messages"] if isinstance(m, HumanMessage)),
        "Unknown question"
    )
    prior      = state.get("reflexion_memory", [])
    prior_text = "\n".join(f"- {l}" for l in prior) if prior else "None yet."

    prompt = reflexion_prompt.format(
        question=question,
        answer=state.get("worker_result", ""),
        score=state.get("quality_score", 0),
        critique=state.get("critique", ""),
        prior_lessons=prior_text,
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    lesson   = response.content.strip().lstrip("-• ").strip()

    print(f"\n[Reflexion] New lesson: {lesson}")

    return {
        "messages":         [AIMessage(content=f"[reflexion]: Lesson learned — {lesson}")],
        "reflexion_memory": [lesson],
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
        "Do not mention workers, critics, reflexion, or internal steps.\n\nHistory:\n"
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

def after_critic(state: AgentState) -> Literal["reflexion_node", "final_node"]:
    score, ref_count = state.get("quality_score", 0), state.get("reflection_count", 0)
    if ref_count >= 3:
        print("[after_critic] Reflection cap → STOP")
        return "final_node"
    if score >= 7:
        print(f"[after_critic] Score {score} ≥ 7 → STOP")
        return "final_node"
    print(f"[after_critic] Score {score} < 7 → Reflexion loop")
    return "reflexion_node"

def after_reflexion(state: AgentState) -> Literal["supervisor", "final_node"]:
    return "final_node" if state.get("reflection_count", 0) >= 3 else "supervisor"

# ─────────────────────────────────────────────
#  Graph assembly
# ─────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("supervisor",     supervisor_node)
graph.add_node("add_node",       add_node)
graph.add_node("multi_node",     multi_node)
graph.add_node("writer_node",    writer_node)
graph.add_node("critic_node",    critic_node)
graph.add_node("reflexion_node", reflexion_node)
graph.add_node("final_node",     final_node)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor",     route_supervisor)
graph.add_conditional_edges("add_node",       should_continue)
graph.add_conditional_edges("multi_node",     should_continue)
graph.add_conditional_edges("writer_node",    should_continue)
graph.add_conditional_edges("critic_node",    after_critic)
graph.add_conditional_edges("reflexion_node", after_reflexion)
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
            "reflexion_memory": [],
        })

        print("\n Final Answer:")
        print(result["messages"][-1].content)

        memory = result.get("reflexion_memory", [])
        if memory:
            print(f"\n Reflexion Memory ({len(memory)} lesson(s)):")
            for i, lesson in enumerate(memory, 1):
                print(f"  {i}. {lesson}")
