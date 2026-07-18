import asyncio
import threading
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


def load_mcp_tools_sync():
    result = {}

    def runner():
        result["tools"] = asyncio.run(get_mcp_tools())

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    return result["tools"]


mcp_tools_list = load_mcp_tools_sync()