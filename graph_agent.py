import os
import json
import asyncio
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent_core import client, MODEL_NAME, tools, TOOL_MAP
from mcp_tools import mcp_tools_list

load_dotenv()

def convert_mcp_tool_to_schema(mcp_tool):
    return {
        "type": "function",
        "function": {
            "name": mcp_tool.name,
            "description": mcp_tool.description,
            "parameters": mcp_tool.args_schema
        }
    }

mcp_tool_schemas = [convert_mcp_tool_to_schema(t) for t in mcp_tools_list]
mcp_tool_map = {t.name: t for t in mcp_tools_list}

all_tools = tools + mcp_tool_schemas


def simple_add(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict):
    messages: Annotated[list, simple_add]
    plan: list


def planner(state: AgentState) -> dict:
    user_question = state["messages"][-1]["content"]

    planning_prompt = [
        {
            "role": "system",
            "content": """You are a planning assistant. Break the user's question into a short, ordered list of concrete research steps needed to answer it fully. Only include steps that require searching for information or performing calculations. Do NOT answer the question yourself. Return ONLY a numbered list, nothing else."""
        },
        {
            "role": "user",
            "content": user_question
        }
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=planning_prompt
    )

    plan_text = response.choices[0].message.content
    plan_steps = [line.strip() for line in plan_text.split("\n") if line.strip()]

    return {"plan": plan_steps}


def call_model(state: AgentState) -> dict:
    plan_context = "\n".join(state.get("plan", []))
    
    system_message = {
        "role": "system",
        "content": f"""You are a helpful research assistant with access to two tools: search_web and calculate. Always search for facts before calculating.

Here is the research plan for this task:
{plan_context}

Follow this plan step by step, using tools as needed. Use your judgment if a step needs adjusting based on what you find."""
    }

    openai_messages = [system_message] + [
        msg for msg in state["messages"] if msg.get("role") != "system"
    ]

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=openai_messages,
        tools=all_tools,
        tool_choice="auto"
    )
    response_message = response.choices[0].message

    assistant_message = {
        "role": "assistant",
        "content": response_message.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            }
            for tc in response_message.tool_calls
        ] if response_message.tool_calls else None
    }

    return {"messages": [assistant_message]}

def execute_tools(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_calls = last_message["tool_calls"]
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["function"]["name"]
        tool_args = json.loads(tool_call["function"]["arguments"])

        if tool_name in TOOL_MAP:
            handler = TOOL_MAP[tool_name]
            result = handler(tool_args)
        elif tool_name in mcp_tool_map:
            mcp_tool = mcp_tool_map[tool_name]
            result = asyncio.run(mcp_tool.ainvoke(tool_args))
        else:
            result = "Tool not found"

        tool_messages.append({
            "role": "tool",
            "tool_call_id": tool_call["id"],
            "content": str(result)
        })

    return {"messages": tool_messages}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    tool_calls = last_message.tool_calls if hasattr(last_message, "tool_calls") else last_message.get("tool_calls")
    if tool_calls:
        return "execute_tools"
    return "END"


def route_entry(state: AgentState) -> str:
    if state.get("plan"):
        return "call_model"
    return "planner"

checkpointer = MemorySaver()
graph = StateGraph(AgentState)

graph.add_node("planner", planner)
graph.add_node("call_model", call_model)
graph.add_node("execute_tools", execute_tools)

graph.set_conditional_entry_point(
    route_entry,
    {
        "planner": "planner",
        "call_model": "call_model"
    }
)

graph.add_edge("planner", "call_model")

graph.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "execute_tools": "execute_tools",
        "END": END
    }
)

graph.add_edge("execute_tools", "call_model")

graph_app = graph.compile(checkpointer=checkpointer)

