import functools
import inspect
from collections.abc import Callable
from typing import Any, get_args, get_origin, get_type_hints

TOOL_REGISTRY: dict[str, Callable[..., Any]] = {}


def _json_schema_for_annotation(annotation: Any) -> dict[str, Any]:
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)
    if origin is list:
        return {"type": "array", "items": _json_schema_for_annotation(get_args(annotation)[0])}
    if origin is dict:
        return {"type": "object"}
    if origin is tuple:
        item_types = get_args(annotation)
        if not item_types:
            return {"type": "array"}
        return {
            "type": "array",
            "items": {"anyOf": [_json_schema_for_annotation(item) for item in item_types]},
        }
    if origin is set:
        return {"type": "array", "items": _json_schema_for_annotation(get_args(annotation)[0])}

    if annotation is str:
        return {"type": "string"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool:
        return {"type": "boolean"}

    try:
        if hasattr(annotation, "__mro__") and annotation.__mro__[0] is not object:
            if annotation.__mro__[0] is bool:
                return {"type": "boolean"}
            if annotation.__mro__[0] in (str, int, float):
                return {"type": _json_schema_for_annotation(annotation.__mro__[0])["type"]}
    except TypeError:
        pass

    return {"type": "string"}


def _tool(
    name: str | None = None, description: str | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        if tool_name in TOOL_REGISTRY:
            raise ValueError(f"A tool named {tool_name!r} is already registered")

        tool_desc = description or inspect.getdoc(func) or ""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties: dict[str, dict[str, Any]] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = type_hints.get(param_name, str)
            schema_for_param = _json_schema_for_annotation(annotation)
            properties[param_name] = schema_for_param

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "name": tool_name,
            "description": tool_desc,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.schema = schema
        wrapper.invoke = lambda kwargs: func(**kwargs)

        return wrapper

    return decorator


def tool(func):
    decorated = _tool()(func)
    TOOL_REGISTRY[decorated.schema["name"]] = decorated
    return decorated
