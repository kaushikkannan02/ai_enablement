# from langchain_aws import ChatBedrock
# from langgraph.prebuilt import ToolNode
# from langchain_core.prompts import ChatPromptTemplate
# from langgraph.graph import StateGraph, END
# from typing import TypedDict, List
# from langchain_core.messages import BaseMessage
# from rag_tool.rag import hr_tool
# from mcp_google_docs.llm_client import insurance_tool

# llm = ChatBedrock(
#     model_id="anthropic.claude-3-sonnet-20240229-v1:0",
#     region_name="us-east-1",
#     model_kwargs={"temperature": 0.2, "max_tokens": 800},
# )

# tools = [hr_tool, insurance_tool]
# llm_with_tools = llm.bind_tools(tools)


# agent_prompt = ChatPromptTemplate.from_messages([
#     ("system", """
# You have access to tools that return detailed information:
# - HRPolicyTool: Returns HR policy information
# - InsurancePolicyTool: Returns insurance policy information

# Instructions:
# 1. When a user asks a question, call the appropriate tool(s).
# 2. The tool will return detailed information - this is the ANSWER, not feedback.
# 3. Present the tool's response directly to the user in a clear format.
# 4. If the tool response is already complete, simply format it nicely for the user.
# 5. Do NOT ask for more questions when you receive tool results - answer based on those results.
# 6. Only use information from the tools, never your own knowledge.
# """),
#     ("human", "{input}")
# ])

# class AgentState(TypedDict):
#     messages: List[BaseMessage]

# def agent_node(state: AgentState):
#     response = llm_with_tools.invoke(
#         agent_prompt.invoke({"input": state["messages"][-1].content})
#     )
#     return {"messages": [response]} 

# tool_node = ToolNode(tools)

# graph = StateGraph(AgentState)
# graph.add_node("agent", agent_node)
# graph.add_node("tools", tool_node)

# graph.set_entry_point("agent")

# graph.add_conditional_edges(
#     "agent",
#     lambda state: "tools" if state["messages"][-1].tool_calls else END,
# )

# graph.add_edge("tools", "agent")

# policy_agent = graph.compile()

from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import BaseTool
from rag_tool.rag import hr_tool
from mcp_google_docs.llm_client import insurance_tool
from mcp_google_docs.web_search import web_search_tool

class SimpleAgent:
    """Simple tool-calling agent with no loops"""
    
    def __init__(self, tools: list[BaseTool]):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        
        self.llm = ChatBedrock(
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            region_name="us-east-1",
            model_kwargs={"temperature": 0.2, "max_tokens": 800},
        )
        
        # Bind tools to LLM
        self.llm_with_tools = self.llm.bind_tools(tools)
        
        self.system_prompt = """You are Presidio's enterprise policy assistant.

You have access to these tools:
- hr_policy_search: Search HR policies (vacation, benefits, work rules, etc.)
- insurance_policy_search: Search insurance policies (coverage, claims, premiums, etc.)
- web_search: search recent trends from the web.

INSTRUCTIONS:
1. Analyze the user's question to determine which tool(s) to use
2. Call the appropriate tool(s) ONCE with the user's question
3. After receiving tool results, present the information clearly to the user
4. DO NOT call tools multiple times or rephrase queries
5. DO NOT add your own knowledge - only use tool results

If a question needs both HR and insurance info, call both tools in one turn.
If a question ever needs the support of web search tool, use web search tool along with the one of the insurance tool or hr tool
"""

    def run(self, user_input: str) -> str:
        """Run a single query through the agent"""
        
        print(f"\n{'='*60}")
        print(f"USER: {user_input}")
        print(f"{'='*60}\n")
        
        # Step 1: Get initial response with tool calls
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_input)
        ]
        
        print("🤖 Agent thinking...")
        response = self.llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Step 2: Execute tools if requested
        if response.tool_calls:
            print(f"🔧 Agent calling {len(response.tool_calls)} tool(s)...\n")
            
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]
                
                print(f"  → {tool_name}({tool_args})")
                
                # Execute the tool
                if tool_name in self.tool_map:
                    try:
                        result = self.tool_map[tool_name].invoke(tool_args)
                        print(f"  ✓ Got result ({len(result)} chars)\n")
                    except Exception as e:
                        result = f"Error calling tool: {str(e)}"
                        print(f"  ✗ Error: {e}\n")
                else:
                    result = f"Tool {tool_name} not found"
                    print("  ✗ Tool not found\n")
                
                # Add tool result to messages
                messages.append(ToolMessage(
                    content=result,
                    tool_call_id=tool_id
                ))
            
            # Step 3: Get final response after tools
            # ✅ FIXED: Use llm_with_tools instead of llm
            print("🤖 Agent synthesizing final answer...")
            final_response = self.llm_with_tools.invoke(messages)
            
            # If agent tries to call tools AGAIN, stop it
            if final_response.tool_calls:
                print("⚠️  Agent tried to call tools again - returning tool results directly")
                # Just return the tool results
                tool_results = "\n\n".join([
                    msg.content for msg in messages 
                    if isinstance(msg, ToolMessage)
                ])
                return tool_results
            
            print(f"\n{'='*60}")
            print(f"ASSISTANT: {final_response.content}")
            print(f"{'='*60}\n")
            
            return final_response.content
        
        else:
            # No tools needed
            print(f"\n{'='*60}")
            print(f"ASSISTANT: {response.content}")
            print(f"{'='*60}\n")
            
            return response.content


# Initialize the agent
policy_agent = SimpleAgent(tools=[hr_tool, insurance_tool, web_search_tool])