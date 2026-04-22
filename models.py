from pydantic import BaseModel, EmailStr
from typing import Optional


class JobDescription(BaseModel):
    id: str
    job_role: str
    description: str


class ResponseCreate(BaseModel):
    user_name: str
    email: str
    job_role: str


class ResponseOut(BaseModel):
    id: str
    user_name: str
    email: str
    job_role: str
    resume_filename: str
    ats_score: Optional[float] = None
