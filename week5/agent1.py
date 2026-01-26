from typing import TypedDict, Literal

from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

# Import your specialist agents
from agent2.agent import run_it_agent
from agent3.agent import run_finance_agent


# -----------------------------
# STATE
# -----------------------------
class SupervisorState(TypedDict):
    user_input: str
    route: Literal["IT", "FINANCE", "UNKNOWN"]
    response: str


# -----------------------------
# LLM (classification only)
# -----------------------------
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0,
        "max_tokens": 50,
    },
)


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
CLASSIFIER_PROMPT = """
You are a supervisor agent.

Your task is to classify the user's request into ONE category:
- IT
- FINANCE
- UNKNOWN

Return ONLY one word: IT, FINANCE, or UNKNOWN.
Do NOT explain your choice.
"""


# -----------------------------
# NODE 1: CLASSIFY
# -----------------------------
def classify_node(state: SupervisorState) -> SupervisorState:
    messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=state["user_input"]),
    ]

    result = llm.invoke(messages).content.strip().upper()

    if result not in {"IT", "FINANCE"}:
        result = "UNKNOWN"

    return {
        **state,
        "route": result,
    }


# -----------------------------
# NODE 2A: ROUTE TO IT
# -----------------------------
def it_node(state: SupervisorState) -> SupervisorState:
    answer = run_it_agent(state["user_input"])
    return {
        **state,
        "response": answer,
    }


# -----------------------------
# NODE 2B: ROUTE TO FINANCE
# -----------------------------
def finance_node(state: SupervisorState) -> SupervisorState:
    answer = run_finance_agent(state["user_input"])
    return {
        **state,
        "response": answer,
    }


# -----------------------------
# NODE 2C: UNKNOWN
# -----------------------------
def unknown_node(state: SupervisorState) -> SupervisorState:
    return {
        **state,
        "response": (
            "I can help with IT or Finance questions only. "
            "Please clarify your request."
        ),
    }


# -----------------------------
# ROUTING LOGIC
# -----------------------------
def route_decision(state: SupervisorState) -> str:
    if state["route"] == "IT":
        return "it"
    if state["route"] == "FINANCE":
        return "finance"
    return "unknown"


# -----------------------------
# BUILD GRAPH
# -----------------------------
graph = StateGraph(SupervisorState)

graph.add_node("classify", classify_node)
graph.add_node("it", it_node)
graph.add_node("finance", finance_node)
graph.add_node("unknown", unknown_node)

graph.set_entry_point("classify")

graph.add_conditional_edges(
    "classify",
    route_decision,
    {
        "it": "it",
        "finance": "finance",
        "unknown": "unknown",
    },
)

graph.add_edge("it", END)
graph.add_edge("finance", END)
graph.add_edge("unknown", END)

agent_1 = graph.compile()


# -----------------------------
# RUN FUNCTION
# -----------------------------
def run_supervisor(user_input: str) -> str:
    state: SupervisorState = {
        "user_input": user_input,
        "route": "UNKNOWN",
        "response": "",
    }

    final_state = agent_1.invoke(state)
    return final_state["response"]


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk Support (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\n--- RESPONSE ---")
        print(run_supervisor(q))
