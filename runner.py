"""Standalone child-process runner for /execute.

Запускается как отдельный процесс из main.py. Это даёт ГАРАНТИРОВАННОЕ
освобождение памяти после исполнения: когда процесс умирает, ОС
забирает всю его RAM (включая то, что налопатили sklearn / numpy /
pandas во время grid search). Основной FastAPI-процесс при этом
остаётся «худым».

Протокол:
- stdin: JSON {"code": "...", "input_data": {...}, "upload_dir": "..."}
- stdout: текст результата (как раньше отдавал /execute)
- exit code: 0 = success, 1 = error в пользовательском коде
"""
from __future__ import annotations

import ast
import datetime as _dt
import io
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any


# =========================================================================
# СЕРИАЛИЗАЦИЯ (копия из main.py — намеренно дублируется, чтобы child
# процесс был независимым и не тащил FastAPI-код)
# =========================================================================

def _to_jsonable(value: Any, _depth: int = 0) -> Any:
    if _depth > 12:
        return repr(value)

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

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

    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _to_plain_text(value: Any) -> str:
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
# ИСПОЛНЕНИЕ
# =========================================================================

def _build_globals(upload_dir: Path, input_data: Any) -> dict:
    def _safe_path(name: str) -> Path:
        safe_name = os.path.basename(name).strip()
        if not safe_name or safe_name in (".", ".."):
            raise ValueError(f"Invalid file name: {name!r}")
        return upload_dir / safe_name

    def _files():
        return sorted(p.name for p in upload_dir.iterdir() if p.is_file())

    def _open_file(name: str, mode: str = "r", encoding="utf-8", **kw):
        path = _safe_path(name)
        if not path.exists():
            raise FileNotFoundError(f"Uploaded file not found: {name}")
        if "b" in mode:
            return open(path, mode, **kw)
        return open(path, mode, encoding=encoding, **kw)

    def _read_file(name: str, encoding: str = "utf-8") -> str:
        return _safe_path(name).read_text(encoding=encoding)

    g: dict = {
        "__builtins__": __builtins__,
        "__name__": "__main__",
        "UPLOAD_DIR": str(upload_dir),
        "files": _files,
        "open_file": _open_file,
        "read_file": _read_file,
        "input_data": input_data,
    }

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
            g[_alias] = __import__(_modname)
        except Exception:
            pass

    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as _plt  # type: ignore
        g["plt"] = _plt
        g["matplotlib"] = matplotlib
    except Exception:
        pass

    return g


def main() -> int:
    raw = sys.stdin.buffer.read()
    try:
        payload = json.loads(raw.decode("utf-8") or "{}")
    except json.JSONDecodeError as e:
        sys.stderr.write(f"runner: invalid input JSON: {e}\n")
        return 2

    code: str = payload.get("code") or ""
    input_data: Any = payload.get("input_data", {}) or {}
    upload_dir = Path(payload.get("upload_dir") or ".")

    try:
        upload_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(upload_dir)
    except Exception:
        pass

    g = _build_globals(upload_dir, input_data)

    real_stdout = sys.stdout
    real_stderr = sys.stderr
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    sys.stdout = stdout_buf
    sys.stderr = stderr_buf

    exit_code = 0
    parts: list[str] = []

    try:
        tree = ast.parse(code, mode="exec")

        last_expr = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last_expr = tree.body.pop()

        if tree.body:
            exec(compile(tree, "<code>", "exec"), g, g)

        if last_expr is not None:
            value = eval(
                compile(ast.Expression(last_expr.value), "<expr>", "eval"),
                g,
                g,
            )
            if value is not None:
                g["result"] = value

        s = stdout_buf.getvalue()
        e = stderr_buf.getvalue()
        if s:
            parts.append(s.rstrip("\n"))
        if e:
            parts.append(e.rstrip("\n"))
        result_text = _to_plain_text(g.get("result"))
        if result_text:
            parts.append(result_text)

    except Exception:
        s = stdout_buf.getvalue()
        e = stderr_buf.getvalue()
        if s:
            parts.append(s.rstrip("\n"))
        if e:
            parts.append(e.rstrip("\n"))
        parts.append(traceback.format_exc().rstrip("\n"))
        exit_code = 1

    finally:
        sys.stdout = real_stdout
        sys.stderr = real_stderr
        try:
            _plt_mod = g.get("plt")
            if _plt_mod is not None:
                _plt_mod.close("all")
        except Exception:
            pass

    real_stdout.write("\n".join(parts))
    real_stdout.flush()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
