import inspect
from functools import wraps
from typing import Literal

from dateutil.relativedelta import relativedelta
from pydantic import validate_call

Horizon = Literal["1mo", "3mo", "6mo", "1y", "2y", "4y"]
Period = Literal["4mo", "1y", "2y", "4y", "8y", "16y"]
DeltaDate = Horizon | Period


@validate_call
def relativedelta_str(delta: DeltaDate) -> relativedelta:
    """Convert a string to a relativedelta object."""
    if delta.endswith("mo"):
        return relativedelta(months=int(delta[:-2]))
    return relativedelta(years=int(delta[:-1]))


def force_list(*args_to_convert):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_args = list(inspect.signature(func).parameters.keys())
            args_map_kwargs = dict(zip(func_args, args, strict=False))
            kwargs.update(args_map_kwargs)

            for key in args_to_convert:
                if key not in func_args:
                    raise ValueError(f"Argument {key} is not a valid argument for {func.__name__}")
                val = kwargs.get(key)
                if val is not None and not isinstance(val, list):
                    kwargs[key] = [val]
            return func(**kwargs)
        return wrapper
    return decorator
