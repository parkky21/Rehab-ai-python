from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Role = Literal["doctor", "patient"]


class RegisterRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    email: str = Field(min_length=5, max_length=200)
    username: str = Field(min_length=3, max_length=60)
    password: str = Field(min_length=8, max_length=128)
    role: Role


class LoginRequest(BaseModel):
    email: str
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserProfile(BaseModel):
    id: str
    name: str
    email: str
    username: str
    role: Role
    created_at: datetime


class PatientAssignmentStats(BaseModel):
    patient: UserProfile
    assigned_count: int
    in_progress_count: int
    completed_count: int
    total_count: int


class PatientLinkRequest(BaseModel):
    patient_id: str | None = None
    patient_email: str | None = None
    patient_username: str | None = None


class AssignmentCreateRequest(BaseModel):
    patient_id: str
    exercise_name: str
    target_reps: int = Field(ge=1, le=200)
    due_date: datetime | None = None
    notes: str | None = Field(default=None, max_length=600)


class AssignmentResponse(BaseModel):
    id: str
    patient_id: str
    doctor_id: str
    exercise_name: str
    target_reps: int
    due_date: datetime | None = None
    status: str
    notes: str | None = None
    created_at: datetime


class ExerciseInfo(BaseModel):
    name: str
    target_rom: float
    ideal_rep_time: float
    acceptable_sway: float


class RealtimeLandmark(BaseModel):
    x: float
    y: float
    z: float
    visibility: float = 1.0


class RealtimeFrameMessage(BaseModel):
    type: Literal["landmark_frame"]
    frame_index: int | None = None
    timestamp_ms: int | None = None
    landmarks: list[RealtimeLandmark]
