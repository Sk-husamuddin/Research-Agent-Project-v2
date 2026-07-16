import uuid
from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI
from agent_core import run_react_loop
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException
from graph_agent import graph_app


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class QueryResponse(BaseModel):
    session_id: str
    answer: Optional[str] = None
    status: str
    last_observation: Optional[str] = None
    last_error: Optional[str] = None

@app.get("/")
def message():
    return {
        "message":"server is running"
    }

@app.post("/query",response_model=QueryResponse)


def query_agent_v2(request: QueryRequest):
    try:
        thread_id = request.session_id or str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        result = graph_app.invoke(
            {"messages": [{"role": "user", "content": request.query}]},
            config=config
        )

        last_message = result["messages"][-1]
        final_answer = last_message["content"] if isinstance(last_message, dict) else last_message.content

        return {
            "session_id": thread_id,
            "answer": final_answer,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent failed: {str(e)}")
