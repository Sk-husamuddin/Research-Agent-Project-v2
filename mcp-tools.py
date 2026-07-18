import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp_client = MultiServerMCPClient(
    {
        "filesystem": {
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                "./agent_workspace"
            ],
            "transport": "stdio"
        }
    }
)

async def get_mcp_tools():
    tools = await mcp_client.get_tools()
    return tools