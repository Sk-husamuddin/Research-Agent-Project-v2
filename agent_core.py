import os
import json
import uuid
import requests
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from asteval import Interpreter
from database.mongo import (
    load_session, save_session, get_cached_result, save_cached_result, save_report
)

load_dotenv()

USE_OPENAI = True

if USE_OPENAI:
    client = OpenAI(api_key=os.getenv("GITHUB_TOKEN"), base_url="https://models.github.ai/inference")
    MODEL_NAME = "gpt-4.1-mini"
else:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    MODEL_NAME = "llama-3.1-8b-instant"

MAX_ITERATIONS = 10
TAVILY_URL = "https://api.tavily.com/search"
TAVILY_MAX_RESULTS = 3
REQUEST_TIMEOUT = 15

aeval = Interpreter()

def calculate(expression: str) -> str:
    try:
        result = aeval(expression)
        if aeval.error:
            return f"Calculation error: {aeval.error[0].get_error()}"
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"


def search_web(query: str) -> str:
    payload = {
        "api_key": os.getenv("TAVILY_API_KEY"),
        "query": query,
        "max_results": TAVILY_MAX_RESULTS,
        "include_answer": True
    }
    try:
        response = requests.post(TAVILY_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(
                f"Title: {item['title']}\n"
                f"URL: {item['url']}\n"
                f"Summary: {item['content']}\n"
            )
        return "\n".join(results)

    except requests.Timeout:
        return "Search error: Request timed out."
    except requests.HTTPError as e:
        return f"Search error: HTTP {e.response.status_code}"
    except requests.RequestException as e:
        return f"Search error: {e}"
    except Exception as e:
        return f"Search error: {str(e)}"


def handle_search_web(tool_args: dict) -> str:
    query = tool_args.get("query")
    if not query:
        return "Tool error: Missing 'query' argument."
    query = query.strip().lower()
    cached = get_cached_result("search_web", query)
    if cached:
        return cached
    result = search_web(query)
    if not result.startswith("Search error"):
        save_cached_result("search_web", query, result)
    return result


def handle_calculate(tool_args: dict) -> str:
    expression = tool_args.get("expression")
    if not expression:
        return "Tool error: Missing 'expression' argument."
    expression = expression.replace(",", "").strip()
    expression = expression.replace("^", "**")
    cached = get_cached_result("calculate", expression)
    if cached:
        return cached
    result = calculate(expression)
    if not result.startswith("Calculation error"):
        save_cached_result("calculate", expression, result)
    return result


tools = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the internet for current, real-time information. Use when the user asks about recent events, live data, or anything not in your training knowledge. Do NOT use for mathematical calculations. Pass ONLY the search query string — no additional fields like date, source, or parameters.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query to look up"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform mathematical calculations. ALWAYS use this tool for any arithmetic, formula, or numerical computation — never calculate in your head. The expression must contain ONLY numbers and operators (e.g. '47 * 89', '1234 + 5678'). NEVER pass words or variable names as the expression — only use this tool after you have the actual numbers from search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "The math expression to evaluate, e.g. '47 * 89'"}
                },
                "required": ["expression"]
            }
        }
    }
]

TOOL_MAP = {
    "search_web": handle_search_web,
    "calculate": handle_calculate
}

def run_react_loop(query: str, session_id: str = None) -> dict:
    if session_id is None:
        session_id = str(uuid.uuid4())

    existing_messages = load_session(session_id)

    if existing_messages:
        messages = existing_messages
        messages.append({"role": "user", "content": query})
    else:
        messages = [
            {
                "role": "system",
                "content": """You are a helpful research assistant with access to two tools:
1. search_web — use for current, real-time information
2. calculate — use for ANY mathematical computation, never compute in your head

Rules:
- Always search for facts BEFORE calculating — never pass words to calculate
- The calculate expression must contain only numbers and operators
- Never add extra fields to tool calls — only pass what the schema requires
- Always cite sources when using search_web
- Use exact numbers from calculate in your final answer, never round
- Always state the direct answer first, then explain if needed. Never describe what tool you used as the answer."""
            },
            {
                "role": "user",
                "content": query
            }
        ]

    iterations = 0

    while iterations < MAX_ITERATIONS:
        iterations += 1
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                tool_choice="auto"
            )
        except Exception as e:
            continue

        response_message = response.choices[0].message

        if response_message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in response_message.tool_calls
                ]
            })

            for tool_call in response_message.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)

                handler = TOOL_MAP.get(tool_name)
                result = handler(tool_args) if handler else "Tool not found"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result)
                })
                save_session(session_id, messages)

        else:
            final_answer = response_message.content
            messages.append({"role": "assistant", "content": final_answer})
            save_session(session_id, messages)
            save_report(session_id=session_id, topic=query, report=final_answer)

            return {
                "session_id": session_id,
                "answer": final_answer,
                "status": "success"
            }

    return {
        "session_id": session_id,
        "answer": None,
        "status": "max_iterations_reached",
        "last_observation": str(messages[-1]["content"])[:300]
    }