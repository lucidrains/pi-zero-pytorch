from typing import List

from pydantic import BaseModel


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


class VideoInfo(BaseModel):
    filename: str
    frames: int
    url: str
    folder: str = ""
