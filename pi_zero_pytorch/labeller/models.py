from typing import List, Optional

from pydantic import BaseModel

# ---- labelling -------------------------------------------------------------

class LabelRequest(BaseModel):
    filename: str
    timestep: int
    success: bool
    penalty: float = -50.0


class InterventionRequest(BaseModel):
    filename: str
    timestep: int


class ActionUpdateRequest(BaseModel):
    filename: str
    timestep: int
    action: List[float]


class FilenameRequest(BaseModel):
    filename: str


class TaskAssignRequest(BaseModel):
    filename: str
    task_id: str

# ---- advantage pipeline ----------------------------------------------------

class AdvantageCalcRequest(BaseModel):
    filename: str
    gamma: float = 0.99
    lam: float = 0.95


class AdvantageStatsRequest(BaseModel):
    percentile: float = 90.0


class BinarizeRequest(BaseModel):
    filename: str
    cutoff: float


class InvalidateRequest(BaseModel):
    filename: str
    cutoff: float = 0.0

# ---- recap workflow --------------------------------------------------------

class RecapTaskRequest(BaseModel):
    task_name: str


class RecapIterRequest(BaseModel):
    task_name: str
    iter_id: int = 0


class LoadDataRequest(BaseModel):
    task_name: Optional[str] = None
    iter_id: Optional[int] = None
    data_id: Optional[str] = None
    is_pretrained: bool = False

# ---- training --------------------------------------------------------------

class ValueTrainRequest(BaseModel):
    config: str = "small"


class PolicyTrainRequest(BaseModel):
    config: str = "mock"
