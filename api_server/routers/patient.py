from bson import ObjectId
from fastapi import APIRouter, Depends
from motor.motor_asyncio import AsyncIOMotorDatabase

from api_server.database import get_db
from api_server.deps import require_role
from api_server.utils import serialize_doc

router = APIRouter(prefix="/patient", tags=["patient"])


def _trend_label(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient_data"
    delta = values[-1] - values[0]
    if delta > 3.0:
        return "improving"
    if delta < -3.0:
        return "declining"
    return "stable"


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


@router.get("/progress")
async def my_progress(
    exercise_name: str | None = None,
    patient: dict = Depends(require_role({"patient"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    session_filter: dict = {"patient_id": ObjectId(patient["id"]), "status": "completed"}
    if exercise_name:
        session_filter["exercise_name"] = exercise_name

    sessions = [serialize_doc(doc) async for doc in db.sessions.find(session_filter).sort("started_at", 1)]
    scores = [float((session.get("summary") or {}).get("avg_final_score", 0.0)) for session in sessions]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    assignments_filter: dict = {"patient_id": ObjectId(patient["id"])}
    if exercise_name:
        assignments_filter["exercise_name"] = exercise_name
    total_assignments = await db.exercise_assignments.count_documents(assignments_filter)
    completed_assignments = await db.exercise_assignments.count_documents({**assignments_filter, "status": "completed"})
    adherence = round((completed_assignments / total_assignments) * 100.0, 1) if total_assignments else 0.0

    progression_filter: dict = {"patient_id": ObjectId(patient["id"])}
    if exercise_name:
        progression_filter["exercise_name"] = exercise_name
    latest_snapshot = await db.progression_snapshots.find_one(progression_filter, sort=[("snapshot_at", -1)])

    return {
        "patient_id": patient["id"],
        "exercise_name": exercise_name,
        "session_count": len(sessions),
        "avg_final_score": avg_score,
        "trend": _trend_label(scores[-5:]),
        "adherence_percent": adherence,
        "recent_scores": scores[-10:],
        "latest_progression": serialize_doc(latest_snapshot),
        "recent_sessions": sessions[-10:],
    }
