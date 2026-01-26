from typing import TypedDict, List

from langchain_aws import ChatBedrock
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, END

# -----------------------------
# IMPORT YOUR TOOLS
# -----------------------------
from .rag_tool import it_tool               # it_internal_policy_search
from .web_search_tool import web_search_tool # external_web_research


# -----------------------------
# STATE DEFINITION
# -----------------------------
class AgentState(TypedDict):
    messages: List[BaseMessage]
    used_internal_tool: bool
    used_web_tool: bool


# -----------------------------
# LLM SETUP
# -----------------------------
llm = ChatBedrock(
    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
    region_name="us-east-1",
    model_kwargs={
        "temperature": 0.2,
        "max_tokens": 800,
    },
)

llm_with_tools = llm.bind_tools([it_tool, web_search_tool])


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are Presidio's IT Support Agent.

Your responsibility is to answer ONLY IT-related questions using the provided tools.
You do NOT handle HR, Finance, or Insurance queries.

You have access to the following tools:
- it_internal_policy_search
- external_web_research

INSTRUCTIONS:
1. Analyze the user's question and confirm it is IT-related.
2. ALWAYS call it_internal_policy_search FIRST.
3. Use external_web_research ONLY if internal documentation does not contain the answer.
4. Each tool may be called AT MOST once.
5. Do NOT rephrase the user's question when calling tools.
6. Do NOT add assumptions or external knowledge.

If the question is not IT-related, clearly say so and stop.
"""


# -----------------------------
# NODE 1: LLM THINKING
# -----------------------------
def llm_node(state: AgentState) -> AgentState:
    response = llm_with_tools.invoke(state["messages"])
    return {
        **state,
        "messages": state["messages"] + [response],
    }


# -----------------------------
# NODE 2: TOOL EXECUTION
# -----------------------------
def tool_node(state: AgentState) -> AgentState:
    last_message = state["messages"][-1]

    if not getattr(last_message, "tool_calls", None):
        return state

    new_messages = state["messages"]

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        # Enforce strict one-time tool usage
        if tool_name == "it_internal_policy_search" and state["used_internal_tool"]:
            continue

        if tool_name == "external_web_research" and state["used_web_tool"]:
            continue

        tool = {
            "it_internal_policy_search": it_tool,
            "external_web_research": web_search_tool,
        }.get(tool_name)

        if not tool:
            continue

        result = tool.invoke(tool_args)

        new_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_id,
            )
        )

        if tool_name == "it_internal_policy_search":
            state["used_internal_tool"] = True
        if tool_name == "external_web_research":
            state["used_web_tool"] = True

    return {
        **state,
        "messages": new_messages,
    }


# -----------------------------
# ROUTING LOGIC
# -----------------------------
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]

    # If the LLM tries to call tools again → STOP
    if getattr(last_message, "tool_calls", None):
        return "tool"

    return END


# -----------------------------
# BUILD GRAPH
# -----------------------------
graph = StateGraph(AgentState)

graph.add_node("llm", llm_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("llm")

graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "tool": "tool",
        END: END,
    },
)

graph.add_edge("tool", "llm")

agent_2 = graph.compile()


# -----------------------------
# RUN FUNCTION
# -----------------------------
def run_it_agent(user_input: str) -> str:
    state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ],
        "used_internal_tool": False,
        "used_web_tool": False,
    }

    final_state = agent_2.invoke(state)

    for msg in reversed(final_state["messages"]):
        if isinstance(msg, BaseMessage) and msg.content:
            return msg.content

    return "No response generated."

