from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback
import sys
import io
import json
from typing import Any, Optional

app = FastAPI(title="Python Runner for n8n")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CodeRequest(BaseModel):
    code: str
    input_data: Optional[Any] = None  # данные из n8n (JSON)


class CodeResponse(BaseModel):
    success: bool
    output: Optional[Any] = None   # результат (возвращается через переменную `result`)
    stdout: str = ""               # всё что напечатал print()
    error: Optional[str] = None


@app.post("/run", response_model=CodeResponse)
async def run_code(request: CodeRequest):
    # Перехватываем stdout
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    local_vars = {
        "data": request.input_data,  # входные данные доступны как `data`
        "result": None,              # результат кладите в `result`
    }

    try:
        exec(request.code, {}, local_vars)
        stdout_value = buffer.getvalue()
        return CodeResponse(
            success=True,
            output=local_vars.get("result"),
            stdout=stdout_value,
        )
    except Exception:
        stdout_value = buffer.getvalue()
        return CodeResponse(
            success=False,
            stdout=stdout_value,
            error=traceback.format_exc(),
        )
    finally:
        sys.stdout = old_stdout


@app.get("/health")
async def health():
    return {"status": "ok"}
