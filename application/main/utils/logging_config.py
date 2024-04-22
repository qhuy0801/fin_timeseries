from typing import Dict, Any

from pydantic import BaseModel, Field


class SweepConfig(BaseModel):
    project: str = Field(default="fin_timeseries")
    method: str = Field(default="bayes")
    metric: Dict[str, Any]
    parameters: Dict[str, Any]
