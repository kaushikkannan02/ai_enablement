from typing import TypedDict, Literal, List

from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from agent2.agent import run_it_agent
from agent3.agent import run_finance_agent


# -----------------------------
# STATE
# -----------------------------
class SupervisorState(TypedDict):
    user_input: str
    route: Literal["IT", "FINANCE", "UNKNOWN"]
    response: str
    trajectory: List[str]
    tools_used: List[str]


# -----------------------------
# LLM (classification only)
# -----------------------------
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    model_kwargs={"temperature": 0, "max_tokens": 50},
)


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
CLASSIFIER_PROMPT = """
You are a supervisor agent.

Classify the user's request into ONE category:
- IT
- FINANCE
- UNKNOWN

Return ONLY one word: IT, FINANCE, or UNKNOWN.
"""


# -----------------------------
# NODE 1: CLASSIFY
# -----------------------------
def classify_node(state: SupervisorState) -> SupervisorState:
    state["trajectory"].append("received_user_input")
    state["tools_used"].append("llm_classifier")

    messages = [
        SystemMessage(content=CLASSIFIER_PROMPT),
        HumanMessage(content=state["user_input"]),
    ]

    result = llm.invoke(messages).content.strip().upper()
    route = result if result in {"IT", "FINANCE"} else "UNKNOWN"

    state["trajectory"].append(f"classified_as_{route.lower()}")

    return {**state, "route": route}


# -----------------------------
# NODE 2A: ROUTE TO IT
# -----------------------------
def it_node(state: SupervisorState) -> SupervisorState:
    state["trajectory"].append("routed_to_it_agent")
    answer = run_it_agent(state["user_input"])
    return {**state, "response": answer}


# -----------------------------
# NODE 2B: ROUTE TO FINANCE
# -----------------------------
def finance_node(state: SupervisorState) -> SupervisorState:
    state["trajectory"].append("routed_to_finance_agent")
    answer = run_finance_agent(state["user_input"])
    return {**state, "response": answer}


# -----------------------------
# NODE 2C: UNKNOWN
# -----------------------------
def unknown_node(state: SupervisorState) -> SupervisorState:
    state["trajectory"].append("routed_to_unknown")
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
    return {
        "IT": "it",
        "FINANCE": "finance",
        "UNKNOWN": "unknown",
    }[state["route"]]


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
# RUN FUNCTION (AgentEval-ready)
# -----------------------------
def run_supervisor(user_input: str, return_trace: bool = False):
    state: SupervisorState = {
        "user_input": user_input,
        "route": "UNKNOWN",
        "response": "",
        "trajectory": [],
        "tools_used":[],
    }

    final_state = agent_1.invoke(state)

    if return_trace:
        return final_state["response"], {
            "trajectory": final_state["trajectory"],
            "route": final_state["route"],
        }

    return final_state["response"]


# -----------------------------
# CLI TEST
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk Support (or 'exit'): ")
        if q.lower() == "exit":
            break

        response, trace = run_supervisor(q, return_trace=True)

        print("\n--- RESPONSE ---")
        print(response)

        print("\n--- TRACE ---")
        for k, v in trace.items():
            print(f"{k}: {v}")





