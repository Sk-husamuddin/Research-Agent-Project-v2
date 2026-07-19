# AI Research Agent V2 (LangGraph + Multi-Agent + MCP)

<!-- Add a hero image or demo GIF here -->
<!-- ![Demo](./assets/demo.gif) -->

## 1. Overview

A LangGraph-based evolution of a hand-built ReAct research agent ([V1](https://github.com/Sk-husamuddin/Research-Agent-Project-v1)), rebuilt to demonstrate production-grade agentic architecture: explicit graph-based orchestration, autonomous multi-step planning, durable session memory via checkpointing, and live integration with an external tool server through the **Model Context Protocol (MCP)**.

Where V1 proved the ReAct pattern by hand-writing the reasoning loop, V2 restructures that same logic as an explicit, inspectable graph — adding a dedicated **planning stage** before execution, and extending the agent's capabilities beyond its own codebase through a standardized, pluggable tool protocol (MCP). Built for the **InnoGenesis 2026** Agentic AI hackathon.

## 2. Motivation

Hand-written control-flow loops (V1's approach) work, but don't scale cleanly to more complex agent behavior — branching logic, multi-step planning, and swapping in new capabilities all require touching the same tangled loop. LangGraph solves this by making the agent's control flow an explicit graph of nodes and edges, which stays comprehensible even as complexity grows.

V2 was built to answer two questions beyond what V1 already proved: **can an agent plan its own multi-step approach before acting**, and **can it gain new capabilities (like file system access) without hand-writing custom integration code for every new tool** — both central to what "agentic AI" means in practice, not just marketing language.

## 3. Features

- **Graph-based orchestration** — the agent's control flow (reason → act → reason...) is an explicit `StateGraph`, not a hidden `while` loop
- **Autonomous planning** — a dedicated Planner node breaks complex, multi-part questions into an ordered research plan *before* any tool execution begins
- **Durable session persistence** — LangGraph checkpointing (`thread_id`-based) automatically reloads full conversation state across separate requests, with zero manual message-stitching
- **Live MCP tool integration** — connected to the official Filesystem MCP server, letting the agent write real research reports to disk through a standardized, pluggable protocol (not custom-coded file I/O)
- **Reused, battle-tested tools** — `search_web` and `calculate` are imported directly from V1's `agent_core.py`, avoiding duplicated logic between versions
- **FastAPI REST API** — same request/response contract as V1, so a frontend can target either version interchangeably

## 4. Architecture

```
┌─────────────┐      HTTP POST /query      ┌──────────────┐
│   Client     │ ─────────────────────────▶ │   FastAPI     │
│ (curl/UI)    │ ◀───────────────────────── │   (app.py)    │
└─────────────┘        JSON response        └──────┬───────┘
                                                     │
                                                     ▼
                                        ┌─────────────────────────┐
                                        │   LangGraph StateGraph   │
                                        │      (graph_agent.py)    │
                                        └─────────────────────────┘
                                                     │
                          ┌──────────────────────────┼──────────────────────────┐
                          ▼                          ▼                          ▼
                 ┌────────────────┐        ┌──────────────────┐        ┌──────────────────┐
                 │    planner      │        │    call_model      │◀──▶│  execute_tools     │
                 │ (runs once per   │──────▶│  (reasons, decides  │    │  (runs requested    │
                 │  new thread)     │        │   next tool call)   │    │   tools, loops back) │
                 └────────────────┘        └──────────────────┘        └──────────────────┘
                                                                                  │
                                                        ┌─────────────────────────┼─────────────────────────┐
                                                        ▼                         ▼                         ▼
                                                ┌───────────────┐        ┌───────────────┐        ┌────────────────────┐
                                                │  search_web    │        │  calculate      │        │  MCP Filesystem      │
                                                │  (Tavily)       │        │  (asteval)      │        │  Server (write files) │
                                                └───────────────┘        └───────────────┘        └────────────────────┘

                                        Persistence: MemorySaver checkpointer, keyed by thread_id
```

## 5. How It Works

1. A user sends a query to `POST /query` with an optional `session_id` (used as LangGraph's `thread_id`).
2. The graph checks its checkpointed state for this `thread_id`. If no plan exists yet (a brand-new thread), execution routes to the **Planner** node first; if a plan already exists (a follow-up message), planning is skipped entirely.
3. **Planner** (new threads only) makes one LLM call to break the question into an ordered list of research steps, stored in graph state.
4. **`call_model`** reads the current plan (if any) as context, then decides the next concrete action — either a tool call or a final answer — informed by the plan but still reasoning per-step, not blindly following a rigid script.
5. If a tool call is requested, **`execute_tools`** runs it — routing to either a local Python function (`search_web`, `calculate`) or, for filesystem operations, bridging into the connected **MCP server** — then loops back to `call_model` with the real result.
6. Once no further tool calls are requested, the graph reaches `END`, and the final answer is returned. The full conversation state — including the plan and every tool result — is automatically checkpointed for that `thread_id`, ready to be reloaded on the next request.

## 6. Tech Stack

| Layer | Technology |
|---|---|
| Language | Python |
| Agent Orchestration | LangGraph (`StateGraph`, conditional edges, checkpointing) |
| Web Framework | FastAPI |
| LLM Providers | Groq (Llama 3.1), GitHub Models (GPT-4.1-mini) — reused from V1 |
| Tool Protocol | Model Context Protocol (MCP) via `langchain-mcp-adapters` |
| MCP Server | Official Filesystem MCP server (`@modelcontextprotocol/server-filesystem`) |
| Web Search | Tavily API |
| Math Evaluation | asteval |
| Persistence | LangGraph `MemorySaver` checkpointer (in-memory, thread-scoped) |

## 7. Folder Structure

```
Research-Agent-Project-v2/
├── agent_core.py          # Reused from V1 — client, tools, TOOL_MAP (not duplicated)
├── graph_agent.py          # LangGraph StateGraph: planner, call_model, execute_tools nodes
├── mcp_tools.py             # MCP client config + async tool-fetching (Filesystem server)
├── app.py                    # FastAPI wrapper — /query endpoint
├── agent_workspace/            # Sandboxed folder the MCP filesystem server can read/write
├── .env.example
└── requirements.txt
```

## 8. Installation

```bash
git clone https://github.com/Sk-husamuddin/Research-Agent-Project-v2.git
cd Research-Agent-Project-v2
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Node.js is required** (the MCP Filesystem server runs via `npx`):
```bash
node --version   # v18+ recommended
```

Create a `.env` file with:
```
GROQ_API_KEY=your_key_here
TAVILY_API_KEY=your_key_here
GITHUB_TOKEN=your_github_models_token
```

Run the API:
```bash
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

## 9. API Endpoints

### `POST /query`

**Request:**
```json
{
  "query": "Search for the current population of Brazil, calculate double that number, and save the result to a file called brazil_report.txt",
  "session_id": null
}
```

**Response:**
```json
{
  "session_id": "generated-or-reused-thread-id",
  "answer": "The current population of Brazil is approximately 219,927,144. Double that is 439,854,288. I've saved this to brazil_report.txt.",
  "status": "success"
}
```

## 10. Agent Workflow

```
                 New thread?
                     │
         ┌───────────┴───────────┐
        Yes                     No
         │                       │
         ▼                       │
     Planner                     │
   (generates plan)              │
         │                       │
         └───────────┬───────────┘
                      ▼
                 call_model ──────────────┐
                      │                   │
                      ▼                   │
            Tool call requested?          │
                      │                   │
                    Yes ──▶ execute_tools ─┘  (loop back with tool result)
                      │
                     No
                      │
                      ▼
              Return Final Answer
```

## 11. Planner Node

Unlike V1 (and V2's own `call_model`), the Planner runs **once per new conversation thread**, not once per message. It reads only the initiating question and produces a short, ordered list of concrete research steps — deliberately *not* answering the question itself. This plan is then injected as context into every subsequent `call_model` call within that thread, guiding — but not rigidly scripting — the agent's step-by-step tool use.

A conditional entry point (`route_entry`) checks whether a plan already exists in the thread's checkpointed state; if so, planning is skipped on follow-up messages, avoiding wasted LLM calls and preventing a planner from generating a nonsensical plan for a context-dependent fragment like "double that number."

## 12. MCP Integration

The agent connects to the official **Filesystem MCP server** via `langchain-mcp-adapters`, restricted to a sandboxed `agent_workspace/` folder. This means the agent can write real files to disk — e.g., saving a completed research report — through a standardized protocol, rather than custom file-handling code specific to this project.

Because MCP communication is inherently asynchronous (the client talks to the server over a subprocess/stdio channel), while the rest of the graph runs synchronously, tool calls are bridged into a dedicated thread at the point of use — avoiding conflicts with FastAPI/uvicorn's own event loop.

This architecture means additional MCP servers (GitHub, databases, Slack, etc.) can be connected later using the same pattern, without rewriting the agent's core reasoning logic.

## 13. Demo

### Example: Multi-step Comparison Query (Planner in Action)

**Request:** *"What is the population of Egypt, and how does it compare to the population of Vietnam? Which one is larger, and by what percentage?"*

<img width="800" alt="Multi-step comparison query" src="https://github.com/user-attachments/assets/75e49340-2d61-4491-b7c3-2d143c6c5f5b" />

**Response:**

<img width="800" alt="Multi-step comparison response" src="https://github.com/user-attachments/assets/bd99a84b-b0cb-4f85-9f86-512894019d98" />

The Planner generated a 4-step plan before any tool was called — visible in the terminal output:

<img width="800" alt="Planner output in terminal" src="https://github.com/user-attachments/assets/56f8ac4d-815c-4c41-8f9f-670f19c55777" />
## 14. Future Improvements

- Durable checkpointing (SQLite or MongoDB-backed) — currently uses in-memory `MemorySaver`, which does not survive a server restart
- Additional MCP servers (GitHub, database) for broader real-world tool access
- Split `execute_tools` into dedicated `search_agent`/`math_agent` nodes with routing, for clearer separation of responsibilities
- Automated agent evaluation suite (accuracy, tool-selection correctness, hallucination checks)
- Vector-based semantic memory (ChromaDB) for cross-session recall

## 15. Learnings

- LangGraph's `add_messages` reducer performs automatic message-format coercion that conflicts with raw OpenAI SDK objects — a custom, non-converting reducer was simpler and more predictable for this use case
- Async MCP clients require careful bridging when embedded inside an already-async host process (FastAPI/uvicorn) — `asyncio.run()` cannot be called from within a running event loop, requiring a dedicated-thread bridge instead
- A Planner is only valuable for genuinely multi-part, multi-step questions; for simple single-fact lookups, it adds latency and cost with no benefit — hence the conditional entry point that skips it on follow-ups
- Reusing V1's `agent_core.py` directly (rather than duplicating tool logic) kept both versions consistent and reduced the surface area for bugs

## 16. License

MIT
