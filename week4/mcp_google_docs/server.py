from fastapi import FastAPI
from pydantic import BaseModel
from tools import TOOLS, run_tool

app = FastAPI(
    title="Presidio MCP – Google Docs (Insurance)",
    description="MCP-compatible server exposing real Google Docs insurance data"
)

@app.get("/tools")
def list_tools():
    return {"tools": list(TOOLS.values())}


class ToolRequest(BaseModel):
    arguments: dict


@app.post("/tools/{tool_name}")
def call_tool(tool_name: str, req: ToolRequest):
    return {
        "result": run_tool(tool_name, req.arguments)
    }
