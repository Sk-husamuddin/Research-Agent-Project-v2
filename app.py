from pydantic import BaseModel
from typing import Optional
from fastapi import FastAPI
from agent_core import run_react_loop
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException


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

def query_agent(request:QueryRequest):
    try:
        result = run_react_loop(query=request.query, session_id=request.session_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500,detail=f"Agent failed:{str(e)}")

