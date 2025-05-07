from dataclasses import dataclass
from typing import Literal

import polars as pl
import polars.selectors as cs

TimesSeriesCol = Literal["Close", "Low", "High", "Close", "Volume", "logR"]


@dataclass(kw_only=True)
class TimesSeriesPolars:
    data: pl.DataFrame
    index_ticker: str | None = None

    def get(
        self, cat: TimesSeriesCol | None = None, include_index: bool = False, include_date: bool = True
    ) -> pl.DataFrame:
        query = cs.all()
        if not include_date:
            query -= cs.date()
        if self.index_ticker is not None and not include_index:
            query -= cs.starts_with(self.index_ticker)
        if cat:
            query -= ~cs.ends_with(f"_{cat}") & ~cs.date()

        result = self.data.select(query).drop_nulls()
        return result.rename({col: col.replace(f"_{cat}", "") for col in result.columns})
