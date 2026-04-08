from bson import ObjectId
from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from api_server.database import get_db
from api_server.deps import require_role
from api_server.utils import serialize_doc

router = APIRouter(prefix="/patient", tags=["patient"])


@router.get("/assignments")
async def my_assignments(
    patient: dict = Depends(require_role({"patient"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    cursor = db.exercise_assignments.find(
        {"patient_id": ObjectId(patient["id"]), "status": {"$in": ["assigned", "in_progress"]}}
    ).sort("created_at", -1)
    assignments = [serialize_doc(doc) async for doc in cursor]
    return {"assignments": assignments}


@router.get("/sessions")
async def my_sessions(
    patient: dict = Depends(require_role({"patient"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    cursor = db.sessions.find({"patient_id": ObjectId(patient["id"])}).sort("started_at", -1).limit(100)
    sessions = [serialize_doc(doc) async for doc in cursor]
    return {"sessions": sessions}
