import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agent_core import client, MODEL_NAME, tools, TOOL_MAP

load_dotenv()


def simple_add(left: list, right: list) -> list:
    return left + right


class AgentState(TypedDict):
    messages: Annotated[list, simple_add]


def call_model(state: AgentState) -> dict:
    openai_messages = []
    for msg in state["messages"]:
        if hasattr(msg, "type"):
            role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
            openai_messages.append({"role": role_map.get(msg.type, msg.type), "content": msg.content})
        else:
            openai_messages.append(msg)

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=openai_messages,
        tools=tools,
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

        handler = TOOL_MAP.get(tool_name)
        result = handler(tool_args) if handler else "Tool not found"

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


graph = StateGraph(AgentState)

graph.add_node("call_model", call_model)
graph.add_node("execute_tools", execute_tools)

graph.set_entry_point("call_model")

graph.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "execute_tools": "execute_tools",
        "END": END
    }
)

graph.add_edge("execute_tools", "call_model")

checkpointer = MemorySaver()
graph_app = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":
    config = {"configurable":{"thread_id":"test-thread-1"}}

    initial_state = {
        "messages": [
            {"role": "system", "content": "You are a helpful research assistant with access to two tools: search_web and calculate. Always search for facts before calculating."},
            {"role": "user", "content": "What is the population of India?"}
        ]
    }

    result = graph_app.invoke(initial_state,config)

    print("FINAL MESSAGES:")
    for msg in result["messages"]:
        print(msg)
    follow_up_state = {
        "messages": [
            {"role": "user", "content": "Double that number."}
        ]
    }

    result2 = graph_app.invoke(follow_up_state,config)

    print("\n=== SECOND RESULT ===")
    for msg in result2["messages"]:
        print(msg)
