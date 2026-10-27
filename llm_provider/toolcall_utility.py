import functools
import inspect
from typing import Callable, Dict, Any, get_type_hints

TOOL_REGISTRY = {}

def _tool(name: str = None, description: str = None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(func: Callable) -> Callable:
        tool_name = name or func.__name__
        tool_desc = description or inspect.getdoc(func) or ""
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        properties = {}
        required = []

        type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean"
        }

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            annotation = type_hints.get(param_name, str)
            json_type = type_mapping.get(annotation, "string")

            properties[param_name] = {
                "type": json_type
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        schema = {
            "name": tool_name,
            "description": tool_desc,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required
            }
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