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