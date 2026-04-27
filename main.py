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
import gc
import ctypes
import ctypes.util
import datetime as _dt
import logging
import asyncio
import subprocess

# Уменьшаем фрагментацию glibc-аллокатора. Без этого RSS у python-процесса
# с numpy/sklearn/pandas раздувается в разы из-за множества arena-куч.
# Должно стоять до импорта numpy и пр., но и сейчас не повредит — это чтение
# при первой malloc-арене. Лучше также задавать в Procfile.
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

# malloc_trim возвращает освобождённую heap-память обратно ОС (glibc, Linux).
# Без этого RSS процесса почти не падает после del/gc.collect().
try:
    _libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
    _malloc_trim = _libc.malloc_trim
    _malloc_trim.argtypes = [ctypes.c_size_t]
    _malloc_trim.restype = ctypes.c_int
except Exception:
    _malloc_trim = None


def _release_memory() -> None:
    """Максимально агрессивное освобождение памяти.

    Несколько проходов GC нужны, потому что pandas/sklearn создают
    циклы со слабыми ссылками — за один проход не всё подбирается.
    malloc_trim(0) после этого возвращает свободные страницы ядру.
    """
    for _ in range(3):
        gc.collect()
    if _malloc_trim is not None:
        try:
            _malloc_trim(0)
        except Exception:
            pass

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

# Лок, сериализующий исполнения /execute (sys.stdout/stderr подменяются
# глобально, поэтому параллельные exec'и в одном процессе делить их не должны).
# Сам exec() выполняется в отдельном потоке через asyncio.to_thread, чтобы
# долгие задачи (например, grid search) не блокировали event loop —
# другие эндпоинты (/upload, /files, /health) остаются отзывчивыми.
_EXECUTE_LOCK: Optional[asyncio.Lock] = None


def _get_execute_lock() -> asyncio.Lock:
    global _EXECUTE_LOCK
    if _EXECUTE_LOCK is None:
        _EXECUTE_LOCK = asyncio.Lock()
    return _EXECUTE_LOCK


def _wipe_upload_dir() -> None:
    """Полностью очищает UPLOAD_DIR от всех файлов и подпапок.
    Вызывается ОДИН РАЗ при старте контейнера/процесса, чтобы между
    перезапусками не копились артефакты прошлых сессий.
    """
    try:
        for _p in UPLOAD_DIR.iterdir():
            try:
                if _p.is_file() or _p.is_symlink():
                    _p.unlink()
                elif _p.is_dir():
                    shutil.rmtree(_p, ignore_errors=True)
            except Exception:
                pass
    except Exception:
        pass


# На старте процесса вычищаем UPLOAD_DIR — на свежем контейнере хранится
# только сам код приложения, никаких остатков от предыдущих запусков.
_wipe_upload_dir()

# Директория для истории (write_history / read_history / delete_history).
# История хранится локально в виде JSON-файлов; при превышении
# HISTORY_MAX_FILES самые старые файлы удаляются автоматически.
HISTORY_DIR = Path(os.environ.get("HISTORY_DIR", "/tmp/python_runner_history"))
HISTORY_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_MAX_FILES = int(os.environ.get("HISTORY_MAX_FILES", 10))

_log = logging.getLogger("python_runner.history")


def _history_files_oldest_first() -> List[Path]:
    """Список JSON-файлов истории, отсортированный по имени (= по дате,
    так как имена содержат timestamp). Самые старые — в начале."""
    return sorted(
        (p for p in HISTORY_DIR.iterdir() if p.is_file() and p.suffix == ".json"),
        key=lambda p: p.name,
    )


def _enforce_history_limit() -> List[str]:
    """Удаляет самые старые файлы из папки истории, пока их > HISTORY_MAX_FILES."""
    files = _history_files_oldest_first()
    deleted: List[str] = []
    while len(files) > HISTORY_MAX_FILES:
        oldest = files.pop(0)
        try:
            oldest.unlink()
            deleted.append(oldest.name)
        except OSError:
            break
    return deleted

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
            head = value.head(1000)
            data = json.loads(head.to_json(orient="records", date_format="iso", default_handler=str))
            res = {
                "__type__": "DataFrame",
                "shape": list(value.shape),
                "columns": [str(c) for c in value.columns],
                "dtypes": {str(k): str(v) for k, v in value.dtypes.items()},
                "data": data,
            }
            del head
            return res
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
# УТИЛИТЫ
# =========================================================================

def _strip_markdown_fence(code: str) -> str:
    """Если код в самом начале и в самом конце обёрнут в markdown-блок
    ```...``` (с языковой меткой или без), возвращает содержимое блока.
    Иначе — исходную строку без изменений (тот же объект).

    «В самом начале/конце» — строго: код начинается с ``` (без ведущих
    пробелов/переводов строк) и заканчивается на ``` (без хвостовых).
    """
    if not code.startswith("```") or not code.endswith("```") or len(code) < 6:
        return code
    first_nl = code.find("\n")
    if first_nl == -1 or first_nl >= len(code) - 3:
        return code
    opening = code[3:first_nl].strip()
    # После ``` допустим только идентификатор языка (буквы/цифры/_/-) или пусто
    if opening and not opening.replace("_", "").replace("-", "").isalnum():
        return code
    return code[first_nl + 1: -3]


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

    deleted: List[str] = []

    # Если файл с таким именем уже существует — затираем его
    if target.exists():
        try:
            target.unlink()
            deleted.append(target.name)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to overwrite existing file: {e}")

    # Освобождаем место под новый файл (без знания точного размера — 0)
    deleted.extend(_ensure_space(required_bytes=0))

    # Стримим содержимое чанками прямо на диск, не держим всё в RAM.
    CHUNK = 1024 * 1024  # 1 MB
    total = 0
    try:
        with open(target, "wb") as out:
            while True:
                chunk = await file.read(CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                total += len(chunk)
                del chunk
    finally:
        try:
            await file.close()
        except Exception:
            pass
    _release_memory()

    return UploadResponse(
        success=True,
        name=target.name,
        path=str(target),
        size=total,
        deleted=deleted,
    )


class Upload2Request(BaseModel):
    url: str
    name: Optional[str] = None


class Upload2Response(BaseModel):
    success: bool
    name: str
    path: str
    size: int
    deleted: List[str] = []
    source_url: str


@app.post("/upload2", response_model=Upload2Response)
async def upload_from_url(req: Upload2Request):
    """Загрузка файла по URL (например, raw GitHub ссылка на CSV).

    Параметры (JSON body):
    - url: прямая ссылка на файл (например https://raw.githubusercontent.com/…/file.csv)
    - name: имя, под которым сохранить (опционально; если не указано — берётся из URL)
    """
    import httpx
    from urllib.parse import urlparse, unquote

    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL is required")

    # Determine target file name
    target_name = req.name
    if not target_name:
        parsed = urlparse(url)
        target_name = unquote(parsed.path.rsplit("/", 1)[-1]) if parsed.path else None
    if not target_name:
        raise HTTPException(status_code=400, detail="Cannot determine file name from URL. Please provide 'name' explicitly.")

    target = _safe_path(target_name)

    deleted: List[str] = []
    if target.exists():
        try:
            target.unlink()
            deleted.append(target.name)
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"Failed to overwrite existing file: {e}")

    deleted.extend(_ensure_space(required_bytes=0))

    # Download the file
    CHUNK = 1024 * 1024  # 1 MB
    total = 0
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=120.0) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download: HTTP {resp.status_code} from {url}",
                    )
                with open(target, "wb") as out:
                    async for chunk in resp.aiter_bytes(chunk_size=CHUNK):
                        out.write(chunk)
                        total += len(chunk)
    except httpx.HTTPError as e:
        # Clean up partial file
        if target.exists():
            target.unlink()
        raise HTTPException(status_code=400, detail=f"Download failed: {e}")
    except HTTPException:
        raise
    except Exception as e:
        if target.exists():
            target.unlink()
        raise HTTPException(status_code=500, detail=f"Unexpected error during download: {e}")

    _release_memory()

    return Upload2Response(
        success=True,
        name=target.name,
        path=str(target),
        size=total,
        deleted=deleted,
        source_url=url,
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

    # Снимаем markdown-обёртку ```python ... ``` (или ``` ... ```), если она
    # стоит строго в самом начале и в самом конце кода. Если обёртка снята —
    # заодно тримим пробелы/переводы строк по краям.
    stripped_code = _strip_markdown_fence(code)
    if stripped_code is not code:
        code = stripped_code.strip()
    if not code.strip():
        raise HTTPException(status_code=400, detail="Empty code")

    # -------------------------
    # 2. RUN IN A SEPARATE PROCESS
    # -------------------------
    # Пользовательский код исполняется в ОТДЕЛЬНОМ дочернем процессе
    # (runner.py). Когда он завершается, ОС забирает всю его RAM —
    # это ГАРАНТИРОВАННОЕ освобождение памяти после grid search и
    # любых других тяжёлых операций. Главный FastAPI-процесс при этом
    # никогда не «толстеет» от пользовательского кода.
    #
    # Лок всё ещё нужен: запускаем по одному child'у одновременно,
    # чтобы они не конкурировали за CPU/RAM на маленьком инстансе.
    async with _get_execute_lock():
        body, status = await _run_in_subprocess(code, input_data)
    # На всякий случай чистим и в родителе (объекты вокруг запроса).
    _release_memory()
    return PlainTextResponse(body, status_code=status)


_RUNNER_SCRIPT = str(Path(__file__).resolve().parent / "runner.py")


async def _run_in_subprocess(code: str, input_data: Any) -> tuple[str, int]:
    """Запускает runner.py в отдельном процессе и возвращает (text, status).

    Передача: stdin — JSON {code, input_data, upload_dir}.
    Получение: stdout — текстовый результат (как раньше).
    Exit code: 0 = ok (HTTP 200), 1 = ошибка пользовательского кода (HTTP 500),
    остальные коды — HTTP 500.
    """
    payload = json.dumps(
        {
            "code": code,
            "input_data": input_data,
            "upload_dir": str(UPLOAD_DIR),
        },
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")

    # Передаём подсказки для уменьшения паразитной памяти в дочернем процессе.
    env = os.environ.copy()
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env.setdefault("OMP_NUM_THREADS", env.get("OMP_NUM_THREADS", "1"))
    env.setdefault("OPENBLAS_NUM_THREADS", env.get("OPENBLAS_NUM_THREADS", "1"))
    env.setdefault("MKL_NUM_THREADS", env.get("MKL_NUM_THREADS", "1"))
    # Не пишем .pyc — экономит небольшой объём диска и память.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-u",  # unbuffered, чтобы stdout приходил сразу
        _RUNNER_SCRIPT,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        cwd=str(UPLOAD_DIR),
    )

    try:
        stdout_b, stderr_b = await proc.communicate(input=payload)
    except Exception as e:
        try:
            proc.kill()
        except Exception:
            pass
        return f"runner failed: {e}", 500

    text = stdout_b.decode("utf-8", errors="replace")
    rc = proc.returncode if proc.returncode is not None else 1

    if rc == 0:
        return text, 200

    # Для отладки приклеиваем stderr дочернего процесса (например, OOM,
    # ImportError при старте интерпретатора и т.д.).
    err = stderr_b.decode("utf-8", errors="replace").rstrip("\n")
    if err:
        text = (text + ("\n" if text else "") + err) if text else err
    return text, 500


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


# Директория с кодом приложения — её НЕ трогаем при wipe-операциях.
APP_DIR = Path(__file__).resolve().parent

# Корни, в которых ищем и удаляем пользовательские артефакты,
# даже если они лежат вне UPLOAD_DIR. Не трогаем системные пути
# (/usr, /bin, /lib, /etc, /proc, /sys, /dev) и саму директорию с кодом.
_WIPE_ROOTS = [
    Path("/tmp"),
    Path("/var/tmp"),
    Path("/root"),
    Path("/data"),
    Path("/workspace"),
    Path(os.path.expanduser("~")),
    UPLOAD_DIR,
]


def _is_protected(path: Path) -> bool:
    """Возвращает True, если путь принадлежит коду приложения и его
    нельзя удалять (сам APP_DIR или что-то внутри него)."""
    try:
        path = path.resolve()
    except Exception:
        return True  # на всякий случай не трогаем
    try:
        path.relative_to(APP_DIR)
        return True
    except ValueError:
        return False


def _wipe_path(p: Path, deleted: List[str], errors: List[dict]) -> None:
    """Удаляет файл/симлинк/директорию, пропуская защищённые пути."""
    try:
        if _is_protected(p):
            return
        if p.is_symlink() or p.is_file():
            p.unlink()
            deleted.append(str(p))
        elif p.is_dir():
            # если внутри лежит APP_DIR — обходим поштучно, иначе rmtree
            try:
                APP_DIR.relative_to(p.resolve())
                contains_app = True
            except ValueError:
                contains_app = False

            if contains_app:
                for child in list(p.iterdir()):
                    _wipe_path(child, deleted, errors)
            else:
                shutil.rmtree(p, ignore_errors=False)
                deleted.append(str(p) + "/")
    except FileNotFoundError:
        pass
    except Exception as e:
        errors.append({"path": str(p), "error": str(e)})


@app.delete("/files")
async def delete_all_files():
    """Удаляет ВСЕ накопленные файлы (загруженные, скачанные, сгенерированные:
    pdf, csv, картинки и т.д.) — как из UPLOAD_DIR, так и из других известных
    рабочих директорий контейнера (/tmp, /var/tmp, /root, /data, /workspace,
    HOME). Код приложения (директория с main.py) не трогается.
    """
    deleted: List[str] = []
    errors: List[dict] = []

    seen: set = set()
    for root in _WIPE_ROOTS:
        try:
            real = root.resolve()
        except Exception:
            continue
        if real in seen or not real.exists() or not real.is_dir():
            continue
        seen.add(real)
        try:
            for child in list(real.iterdir()):
                _wipe_path(child, deleted, errors)
        except Exception as e:
            errors.append({"path": str(real), "error": str(e)})

    # Пересоздаём UPLOAD_DIR, чтобы дальнейшие /upload и /execute работали.
    try:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append({"path": str(UPLOAD_DIR), "error": f"recreate failed: {e}"})

    _release_memory()
    return {
        "success": True,
        "upload_dir": str(UPLOAD_DIR),
        "app_dir": str(APP_DIR),
        "roots": [str(r) for r in seen],
        "deleted": deleted,
        "count": len(deleted),
        "errors": errors,
    }


# =========================================================================
# HISTORY ENDPOINTS  (хранение на Яндекс Диске)
# =========================================================================

_MSK_TZ = _dt.timezone(_dt.timedelta(hours=3))


@app.post("/write_history")
async def write_history(request: Request):
    """Принимает JSON и сохраняет его как файл в локальной папке HISTORY_DIR.

    Имя файла = текущая дата-время по Москве (MSK).
    В папке хранится не более HISTORY_MAX_FILES (по умолчанию 10) файлов —
    самые старые удаляются автоматически.
    """
    raw_body = await request.body()
    try:
        payload = json.loads(raw_body.decode("utf-8") or "null")
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    now_msk = _dt.datetime.now(_MSK_TZ)
    # Имя файла: 2026-04-26_21-32-23-123456.json
    name = now_msk.strftime("%Y-%m-%d_%H-%M-%S-") + f"{now_msk.microsecond:06d}.json"
    target = HISTORY_DIR / name

    try:
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Failed to write history: {e}")

    # Удаляем старые файлы, если превышен лимит
    try:
        deleted = _enforce_history_limit()
    except Exception:
        deleted = []

    return {
        "success": True,
        "name": name,
        "path": str(target),
        "deleted": deleted,
    }


@app.get("/read_history")
async def read_history():
    """Возвращает JSON вида {имя файла: содержание (как JSON)}.

    Файлы читаются из локальной папки ``HISTORY_DIR``.
    """
    result: dict = {}
    for p in _history_files_oldest_first():
        try:
            with open(p, "r", encoding="utf-8") as f:
                result[p.name] = json.load(f)
        except Exception as e:
            result[p.name] = {"__error__": f"Failed to read: {e}"}
    return result


@app.delete("/delete_history")
async def delete_history():
    """Удаляет всю историю — все файлы из папки HISTORY_DIR."""
    deleted: List[str] = []
    for p in _history_files_oldest_first():
        try:
            p.unlink()
            deleted.append(p.name)
        except Exception:
            pass
    return {"success": True, "deleted": deleted, "count": len(deleted)}


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
