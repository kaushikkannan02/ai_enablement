# from langchain_aws import ChatBedrock
# from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
# from langchain_core.tools import BaseTool
# from .rag_tool import finance_tool
# from .web_search_tool import web_search_tool

# class SimpleAgent:
#     """Simple tool-calling agent with no loops"""
    
#     def __init__(self, tools: list[BaseTool]):
#         self.tools = tools
#         self.tool_map = {tool.name: tool for tool in tools}
        
#         self.llm = ChatBedrock(
#             model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#             region_name="us-east-1",
#             model_kwargs={"temperature": 0.2, "max_tokens": 800},
#         )
        
#         # Bind tools to LLM
#         self.llm_with_tools = self.llm.bind_tools(tools)
        
#         self.system_prompt = """
# You are Presidio's Finance Support Agent.

# Your responsibility is to answer ONLY Finance-related questions using the provided tools.
# You do NOT handle IT, HR, or Insurance queries.

# You have access to the following tools:
# - finance_internal_policy_search: Retrieve official internal Finance policies, procedures, and reports.
# - external_web_research: Retrieve authoritative external Finance, Accounting, or Compliance information when internal documentation is insufficient.

# INSTRUCTIONS:
# 1. Analyze the user's question and confirm it is Finance-related.
# 2. ALWAYS call finance_internal_policy_search FIRST with the user's question.
# 3. Use external_web_research ONLY if the internal Finance documentation does not contain the required information.
# 4. Call each tool at most ONCE per question.
# 5. If both tools are used, call them in the same turn.
# 6. Do NOT rephrase or modify the user's question when calling tools.
# 7. Do NOT add assumptions, opinions, or knowledge beyond tool outputs.
# 8. If internal documentation contradicts external sources, ALWAYS follow internal documentation.

# After receiving tool results:
# - Present the information clearly and concisely.
# - Use only the information returned by the tools.
# - If the information is unavailable, state this explicitly.

# STRICT RULES:
# - Do NOT answer IT, HR, or Insurance questions.
# - Do NOT call tools multiple times.
# - Do NOT hallucinate or infer missing details.
# """


#     def run(self, user_input: str) -> str:
#         """Run a single query through the agent"""
        
#         print(f"\n{'='*60}")
#         print(f"USER: {user_input}")
#         print(f"{'='*60}\n")
        
#         # Step 1: Get initial response with tool calls
#         messages = [
#             SystemMessage(content=self.system_prompt),
#             HumanMessage(content=user_input)
#         ]
        
#         print("🤖 Agent thinking...")
#         response = self.llm_with_tools.invoke(messages)
#         messages.append(response)
        
#         # Step 2: Execute tools if requested
#         if response.tool_calls:
#             print(f"🔧 Agent calling {len(response.tool_calls)} tool(s)...\n")
            
#             for tool_call in response.tool_calls:
#                 tool_name = tool_call["name"]
#                 tool_args = tool_call["args"]
#                 tool_id = tool_call["id"]
                
#                 print(f"  → {tool_name}({tool_args})")
                
#                 # Execute the tool
#                 if tool_name in self.tool_map:
#                     try:
#                         result = self.tool_map[tool_name].invoke(tool_args)
#                         print(f"  ✓ Got result ({len(result)} chars)\n")
#                     except Exception as e:
#                         result = f"Error calling tool: {str(e)}"
#                         print(f"  ✗ Error: {e}\n")
#                 else:
#                     result = f"Tool {tool_name} not found"
#                     print("  ✗ Tool not found\n")
                
#                 # Add tool result to messages
#                 messages.append(ToolMessage(
#                     content=result,
#                     tool_call_id=tool_id
#                 ))
            
#             # Step 3: Get final response after tools
#             # ✅ FIXED: Use llm_with_tools instead of llm
#             print("🤖 Agent synthesizing final answer...")
#             final_response = self.llm_with_tools.invoke(messages)
            
#             # If agent tries to call tools AGAIN, stop it
#             if final_response.tool_calls:
#                 print("⚠️  Agent tried to call tools again - returning tool results directly")
#                 # Just return the tool results
#                 tool_results = "\n\n".join([
#                     msg.content for msg in messages 
#                     if isinstance(msg, ToolMessage)
#                 ])
#                 print(f"\n{'='*60}")
#                 print(f"ASSISTANT: {tool_results}")
#                 print(f"{'='*60}\n")
            
#             print(f"\n{'='*60}")
#             print(f"ASSISTANT: {final_response.content}")
#             print(f"{'='*60}\n")
            
#             return final_response.content
        
#         else:
#             # No tools needed
#             print(f"\n{'='*60}")
#             print(f"ASSISTANT: {response.content}")
#             print(f"{'='*60}\n")
            
#             return response.content


# # Initialize the agent
# agent_3 = SimpleAgent(tools=[finance_tool, web_search_tool])

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
# IMPORT FINANCE TOOLS
# -----------------------------
from .rag_tool import finance_tool              # finance_internal_policy_search
from .web_search_tool import web_search_tool     # external_web_research


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

llm_with_tools = llm.bind_tools([finance_tool, web_search_tool])


# -----------------------------
# SYSTEM PROMPT
# -----------------------------
SYSTEM_PROMPT = """
You are Presidio's Finance Support Agent.

Your responsibility is to answer ONLY Finance-related questions using the provided tools.
You do NOT handle IT, HR, or Insurance queries.

You have access to the following tools:
- finance_internal_policy_search
- external_web_research

INSTRUCTIONS:
1. Analyze the user's question and confirm it is Finance-related.
2. ALWAYS call finance_internal_policy_search FIRST.
3. Use external_web_research ONLY if internal documentation does not contain the answer.
4. Each tool may be called AT MOST once.
5. Do NOT rephrase the user's question when calling tools.
6. Do NOT add assumptions or external knowledge.

If the question is not Finance-related, clearly say so and stop.
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

        # Enforce strict one-time usage
        if tool_name == "finance_internal_policy_search" and state["used_internal_tool"]:
            continue

        if tool_name == "external_web_research" and state["used_web_tool"]:
            continue

        tool = {
            "finance_internal_policy_search": finance_tool,
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

        if tool_name == "finance_internal_policy_search":
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

    # Prevent second tool invocation
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

agent_3 = graph.compile()


# -----------------------------
# RUN FUNCTION
# -----------------------------
def run_finance_agent(user_input: str) -> str:
    state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input),
        ],
        "used_internal_tool": False,
        "used_web_tool": False,
    }

    final_state = agent_3.invoke(state)

    for msg in reversed(final_state["messages"]):
        if isinstance(msg, BaseMessage) and msg.content:
            return msg.content

    return "No response generated."

