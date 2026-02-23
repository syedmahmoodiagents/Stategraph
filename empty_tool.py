import os
from dotenv import load_dotenv
load_dotenv()
os.getenv("HF_TOKEN")

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.tools import tool
# from langchain.agents import create_agent
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from langchain_core.messages import HumanMessage

llm = ChatHuggingFace(llm = HuggingFaceEndpoint(repo_id='openai/gpt-oss-20b'))

def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b

worker_add = create_react_agent(
    llm,
    tools=[add],   
    name="add_worker"
)

worker_multi = create_react_agent(
    llm,
    tools=[multiply],
    name="multi_worker"
)

worker_writer = create_react_agent(
    llm,
    tools=[],
    name="writer_worker"
)

supervisor = create_supervisor(
    model=llm,
    agents=[worker_add, worker_multi, worker_writer],
    name="supervisor_agent",
    prompt="if it addition or multiplication task, assign it to add_worker or multi_worker. if it is a writing task, assign it to writer_worker.",
)

app = supervisor.compile()

response1 = app.invoke(
    
    {
        "messages": [HumanMessage(content="who is the president of United States ?")],
    }
)

print("Math Task Output:\n")
print(response1["messages"][-1].content)
