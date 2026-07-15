import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from agent_core import client, MODEL_NAME, tools, TOOL_MAP

load_dotenv()


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def call_model(state: AgentState) -> dict:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=state["messages"],
        tools=tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message
    return {"messages": [response_message]}

def execute_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)

        handler = TOOL_MAP.get(tool_name)
        result = handler(tool_args) if handler else "Tool not found"

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result)
        })

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "execute_tools"
    return "END"