from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, PlainTextResponse
from pydantic import BaseModel
import traceback
import sys
import io
import os
import json
import shutil
from pathlib import Path
from typing import Any, Optional, List
import math
import datetime as _dt

app = FastAPI(
    title="Python Runner for n8n",
    description="Сервис для загрузки файлов и исполнения Python-кода (jupyter-like).",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Директория для загруженных файлов
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "/tmp/python_runner_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Минимум свободного места на диске (в байтах). По умолчанию 100 МБ.
MIN_FREE_BYTES = int(os.environ.get("MIN_FREE_BYTES", 100 * 1024 * 1024))
# Максимальный суммарный размер директории загрузок (в байтах). 0 = без лимита. По умолчанию 1 ГБ.
MAX_UPLOAD_DIR_BYTES = int(os.environ.get("MAX_UPLOAD_DIR_BYTES", 1024 * 1024 * 1024))


def _dir_size() -> int:
    return sum(p.stat().st_size for p in UPLOAD_DIR.iterdir() if p.is_file())


def _free_space() -> int:
    try:
        return shutil.disk_usage(UPLOAD_DIR).free
    except Exception:
        return -1


def _files_oldest_first() -> List[Path]:
    return sorted(
        (p for p in UPLOAD_DIR.iterdir() if p.is_file()),
        key=lambda p: p.stat().st_mtime,
    )


def _ensure_space(required_bytes: int = 0) -> List[str]:
    """Удаляет старые файлы, пока:
    - свободного места меньше MIN_FREE_BYTES + required_bytes, или
    - суммарный размер директории больше MAX_UPLOAD_DIR_BYTES (если задан).
    Возвращает список удалённых файлов.
    """
    deleted: List[str] = []
    files = _files_oldest_first()

    def need_cleanup() -> bool:
        free = _free_space()
        if free >= 0 and free < (MIN_FREE_BYTES + required_bytes):
            return True
        if MAX_UPLOAD_DIR_BYTES > 0 and _dir_size() + required_bytes > MAX_UPLOAD_DIR_BYTES:
            return True
        return False

    while files and need_cleanup():
        oldest = files.pop(0)
        try:
            oldest.unlink()
            deleted.append(oldest.name)
        except OSError:
            break
    return deleted


def _safe_path(name: str) -> Path:
    """Защита от path traversal: оставляем только basename."""
    safe_name = os.path.basename(name).strip()
    if not safe_name or safe_name in (".", ".."):
        raise HTTPException(status_code=400, detail="Invalid file name")
    return UPLOAD_DIR / safe_name


# =========================================================================
# СЕРИАЛИЗАЦИЯ РЕЗУЛЬТАТОВ (jupyter-like, но JSON-friendly)
# =========================================================================

def _to_jsonable(value: Any, _depth: int = 0) -> Any:
    """Рекурсивно превращает значение в JSON-сериализуемое представление,
    сохраняя структуру (а не сводя всё к строкам).

    Поддерживает: pandas.DataFrame / Series / Index, numpy скаляры и массивы,
    dtype-объекты, datetime/Timestamp, Path, set/tuple и пр.
    """
    if _depth > 12:
        return repr(value)

    # быстрые случаи
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    # pandas / numpy — импортируем лениво
    try:
        import pandas as pd  # type: ignore
    except Exception:
        pd = None  # type: ignore
    try:
        import numpy as np  # type: ignore
    except Exception:
        np = None  # type: ignore

    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return {
                "__type__": "DataFrame",
                "shape": list(value.shape),
                "columns": [str(c) for c in value.columns],
                "dtypes": {str(k): str(v) for k, v in value.dtypes.items()},
                "data": json.loads(value.head(1000).to_json(orient="records", date_format="iso", default_handler=str)),
            }
        if isinstance(value, pd.Series):
            # включая результат df.dtypes (Series of dtype-объектов)
            return {str(k): _to_jsonable(v, _depth + 1) for k, v in value.items()}
        if isinstance(value, pd.Index):
            return [_to_jsonable(v, _depth + 1) for v in value.tolist()]
        if isinstance(value, pd.Timestamp):
            return value.isoformat()

    if np is not None:
        if isinstance(value, np.dtype):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return [_to_jsonable(v, _depth + 1) for v in value.tolist()]

    # numpy/pandas dtype может прилететь и не как np.dtype
    if type(value).__name__ == "dtype":
        return str(value)

    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, _dt.timedelta):
        return value.total_seconds()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")
    if isinstance(value, (set, frozenset, tuple)):
        return [_to_jsonable(v, _depth + 1) for v in value]
    if isinstance(value, list):
        return [_to_jsonable(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, _depth + 1) for k, v in value.items()}

    # fallback — пытаемся через json, иначе str
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _to_plain_text(value: Any) -> str:
    """Преобразует результат в plain text.

    - None -> ""
    - str -> сам себя
    - pandas.DataFrame -> человекочитаемая таблица (df.to_string())
    - pandas.Series -> series.to_string()
    - bytes -> декод UTF-8
    - dict / list / прочее -> JSON (через _to_jsonable) с отступами,
      а если не сериализуется — str(value)
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="replace")

    try:
        import pandas as pd  # type: ignore
        if isinstance(value, pd.DataFrame):
            return value.to_string(index=False)
        if isinstance(value, pd.Series):
            return value.to_string()
    except Exception:
        pass

    if isinstance(value, (dict, list, tuple, set, frozenset)):
        try:
            return json.dumps(_to_jsonable(value), ensure_ascii=False, indent=2, default=str)
        except Exception:
            return str(value)

    return str(value)


# =========================================================================
# НОВЫЕ ЭНДПОИНТЫ
# =========================================================================

class UploadResponse(BaseModel):
    success: bool
    name: str
    path: str
    size: int
    deleted: List[str] = []


class ExecuteRequest(BaseModel):
    code: str


class ExecuteResponse(BaseModel):
    success: bool
    stdout: str = ""
    stderr: str = ""
    result: str = ""                # значение последнего выражения (как в jupyter), как plain text
    error: Optional[str] = None
    files: List[str] = []           # список доступных файлов


@app.post("/upload", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """Загрузка текстового файла. Параметры:
    - file: содержимое
    - name: имя, под которым сохранить (опционально, иначе имя из file.filename)
    """
    target_name = name or file.filename
    if not target_name:
        raise HTTPException(status_code=400, detail="File name is required")
    target = _safe_path(target_name)

    content = await file.read()

    deleted: List[str] = []

    # Если файл с таким именем уже существует — затираем его
    if target.exists():
        try:
            target.unlink()
            deleted.append(target.name)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to overwrite existing file: {e}")

    # Освобождаем место под новый файл, удаляя старые при необходимости
    deleted.extend(_ensure_space(required_bytes=len(content)))

    target.write_bytes(content)

    return UploadResponse(
        success=True,
        name=target.name,
        path=str(target),
        size=len(content),
        deleted=deleted,
    )


@app.post("/execute", response_class=PlainTextResponse)
async def execute_code(request: Request):
    """Исполнение python-кода с доступом к ранее загруженным файлам.

    Эндпоинт максимально терпим к формату тела запроса — пиши как удобно:
    - **raw / text/plain**: тело целиком трактуется как python-код. Самый
      простой вариант: ничего экранировать не надо, переносы строк сохраняются.
    - **application/json**: `{"code": "...", "input_data": {...}}` — поля
      `code` и `input_data` берутся из JSON-объекта.
    - **application/x-www-form-urlencoded** / **multipart/form-data**: поле `code`,
      остальные поля доступны в `input_data`.

    Любой другой Content-Type (или вообще без него) — обрабатывается как raw.
    Если raw-тело является JSON-объектом с полем `code`, оно тоже будет распознано.

    В контексте кода доступны:
    - UPLOAD_DIR: путь к директории с загруженными файлами (str)
    - files(): список имён загруженных файлов
    - open_file(name, mode='r', ...): открытие загруженного файла
    - read_file(name, encoding='utf-8'): прочитать файл целиком как текст
    - input_data: данные из тела запроса (dict)

    Результат как в jupyter: значение последнего выражения возвращается в `result`.
    Также можно явно присвоить переменную `result`.
    """
    import ast

    content_type = (request.headers.get("content-type") or "").lower()
    raw_body = await request.body()

    code: Optional[str] = None
    input_data: Any = {}

    # -------------------------
    # 1. PARSE REQUEST
    # -------------------------
    if "application/json" in content_type:
        try:
            payload = json.loads(raw_body.decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

        if isinstance(payload, dict):
            code = payload.get("code")
            input_data = payload.get("input_data", {})

    elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
        form = await request.form()
        code = form.get("code")
        input_data = dict(form)

    else:
        body_text = raw_body.decode("utf-8", errors="replace")
        stripped = body_text.lstrip()
        if stripped.startswith("{"):
            # похоже на JSON — пробуем разобрать как {"code": "...", "input_data": {...}}
            try:
                payload = json.loads(body_text)
                if isinstance(payload, dict):
                    if isinstance(payload.get("code"), str):
                        code = payload["code"]
                    input_data = payload.get("input_data", {}) or {}
            except json.JSONDecodeError:
                pass
        if code is None:
            # raw body как код «как есть»
            code = body_text

    if not isinstance(code, str) or not code.strip():
        raise HTTPException(status_code=400, detail="Empty code")

    # -------------------------
    # 2. STDOUT CAPTURE
    # -------------------------
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = stdout_buf = io.StringIO()
    sys.stderr = stderr_buf = io.StringIO()

    # -------------------------
    # 3. FILE HELPERS (IMPORTANT FIX)
    # -------------------------
    def _files():
        return sorted(p.name for p in UPLOAD_DIR.iterdir() if p.is_file())

    def _open_file(name: str, mode: str = "r", encoding: Optional[str] = "utf-8", **kw):
        path = _safe_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {name}")
        if "b" in mode:
            return open(path, mode, **kw)
        return open(path, mode, encoding=encoding, **kw)

    def _read_file(name: str, encoding: str = "utf-8") -> str:
        return _safe_path(name).read_text(encoding=encoding)

    # -------------------------
    # 4. EXEC CONTEXT (FIXED)
    # -------------------------
    globals_dict = {
        "__builtins__": __builtins__,
        "UPLOAD_DIR": str(UPLOAD_DIR),
        "files": _files,
        "open_file": _open_file,
        "read_file": _read_file,
        "input_data": input_data,
    }

    # Преимпортируем популярные библиотеки, чтобы пользовательский код мог
    # сразу писать `pd.read_csv(...)` / `np.array(...)` без явного import.
    # Если пакет не установлен — просто пропускаем.
    for _alias, _modname in (
        ("pd", "pandas"),
        ("np", "numpy"),
        ("json", "json"),
        ("math", "math"),
        ("os", "os"),
        ("re", "re"),
        ("datetime", "datetime"),
    ):
        try:
            globals_dict[_alias] = __import__(_modname)
        except Exception:
            pass

    # matplotlib.pyplot — отдельно, с headless-бэкендом (на сервере нет дисплея).
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt  # type: ignore
        globals_dict["plt"] = _plt
        globals_dict["matplotlib"] = matplotlib
    except Exception:
        pass

    # Чтобы относительные пути в коде (например, pd.read_csv("mini.csv"))
    # резолвились в UPLOAD_DIR, временно меняем cwd на время исполнения.
    _prev_cwd = os.getcwd()
    try:
        os.chdir(UPLOAD_DIR)
    except Exception:
        pass

    try:
        tree = ast.parse(code, mode="exec")

        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()

        if tree.body:
            exec(compile(tree, "<code>", "exec"), globals_dict, globals_dict)

        if last_expr is not None:
            value = eval(
                compile(ast.Expression(last_expr.value), "<expr>", "eval"),
                globals_dict,
                globals_dict,
            )
            if value is not None:
                globals_dict["result"] = value

        parts: List[str] = []
        stdout_value = stdout_buf.getvalue()
        stderr_value = stderr_buf.getvalue()
        if stdout_value:
            parts.append(stdout_value.rstrip("\n"))
        if stderr_value:
            parts.append(stderr_value.rstrip("\n"))
        result_text = _to_plain_text(globals_dict.get("result"))
        if result_text:
            parts.append(result_text)
        return PlainTextResponse("\n".join(parts))

    except Exception:
        parts = []
        stdout_value = stdout_buf.getvalue()
        stderr_value = stderr_buf.getvalue()
        if stdout_value:
            parts.append(stdout_value.rstrip("\n"))
        if stderr_value:
            parts.append(stderr_value.rstrip("\n"))
        parts.append(traceback.format_exc().rstrip("\n"))
        return PlainTextResponse("\n".join(parts), status_code=500)

    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        try:
            os.chdir(_prev_cwd)
        except Exception:
            pass


@app.get("/files")
async def list_files():
    """Список загруженных файлов."""
    return {
        "upload_dir": str(UPLOAD_DIR),
        "files": sorted(p.name for p in UPLOAD_DIR.iterdir() if p.is_file()),
    }


@app.delete("/files/{name}")
async def delete_file(name: str):
    path = _safe_path(name)
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    path.unlink()
    return {"success": True, "name": name}


# =========================================================================
# LEGACY (оставлено для обратной совместимости, не использовать в новом коде)
# =========================================================================

class CodeRequest(BaseModel):
    """[LEGACY] Используйте /execute."""
    code: str
    input_data: Optional[Any] = None  # данные из n8n (JSON)


class CodeResponse(BaseModel):
    """[LEGACY] Используйте /execute."""
    success: bool
    output: Optional[Any] = None   # результат (возвращается через переменную `result`)
    stdout: str = ""               # всё что напечатал print()
    error: Optional[str] = None


@app.post("/run", response_model=CodeResponse, deprecated=True, tags=["legacy"])
async def run_code(request: CodeRequest):
    """[LEGACY] Старый эндпоинт исполнения кода. Используйте POST /execute."""
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()

    local_vars = {
        "data": request.input_data,
        "input_data": request.input_data,  # ФИX: теперь input_data тоже доступна, как в /execute
        "result": None,
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
