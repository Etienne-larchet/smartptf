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
