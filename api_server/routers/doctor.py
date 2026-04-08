from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api_server.database import get_db
from api_server.deps import require_role
from api_server.exercise_factory import list_exercises
from api_server.schemas import (
    AssignmentCreateRequest,
    AssignmentResponse,
    PatientAssignmentStats,
    PatientLinkRequest,
    UserProfile,
)
from api_server.utils import serialize_doc, to_object_id, utc_now

router = APIRouter(prefix="/doctor", tags=["doctor"])


def _public_patient(patient_doc: dict) -> dict:
    return {
        "id": str(patient_doc["_id"]),
        "name": patient_doc.get("name"),
        "email": patient_doc.get("email"),
        "username": patient_doc.get("username"),
        "role": patient_doc.get("role"),
        "created_at": patient_doc.get("created_at"),
    }


def _trend_label(values: list[float]) -> str:
    if len(values) < 2:
        return "insufficient_data"
    delta = values[-1] - values[0]
    if delta > 3.0:
        return "improving"
    if delta < -3.0:
        return "declining"
    return "stable"


async def _assert_linked(db: AsyncIOMotorDatabase, doctor_id: str, patient_id: ObjectId) -> None:
    linked = await db.doctor_patient_links.find_one(
        {
            "doctor_id": ObjectId(doctor_id),
            "patient_id": patient_id,
            "status": "active",
        }
    )
    if not linked:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Patient is not linked to this doctor")


@router.post("/patients/link")
async def link_patient(
    payload: PatientLinkRequest,
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    patient_doc = None

    if payload.patient_id:
        patient_id = to_object_id(payload.patient_id, "patient_id")
        patient_doc = await db.users.find_one({"_id": patient_id, "role": "patient"})
    elif payload.patient_email:
        patient_doc = await db.users.find_one(
            {"email": payload.patient_email.strip().lower(), "role": "patient"}
        )
    elif payload.patient_username:
        patient_doc = await db.users.find_one(
            {"username": payload.patient_username.strip().lower(), "role": "patient"}
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Provide one of patient_id, patient_email, or patient_username",
        )

    if not patient_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

    patient_id = patient_doc["_id"]

    link_doc = {
        "doctor_id": ObjectId(doctor["id"]),
        "patient_id": patient_id,
        "status": "active",
        "created_at": utc_now(),
    }
    await db.doctor_patient_links.update_one(
        {"doctor_id": link_doc["doctor_id"], "patient_id": link_doc["patient_id"]},
        {"$setOnInsert": link_doc},
        upsert=True,
    )

    return {
        "status": "linked",
        "patient_id": str(patient_id),
        "patient_email": patient_doc.get("email"),
        "patient_username": patient_doc.get("username"),
    }


@router.post("/assignments", response_model=AssignmentResponse)
async def create_assignment(
    payload: AssignmentCreateRequest,
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> AssignmentResponse:
    if payload.exercise_name not in list_exercises():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported exercise")

    patient_id = to_object_id(payload.patient_id, "patient_id")

    await _assert_linked(db, doctor["id"], patient_id)

    assignment_doc = {
        "doctor_id": ObjectId(doctor["id"]),
        "patient_id": patient_id,
        "exercise_name": payload.exercise_name,
        "target_reps": payload.target_reps,
        "due_date": payload.due_date,
        "notes": payload.notes,
        "status": "assigned",
        "created_at": utc_now(),
    }
    result = await db.exercise_assignments.insert_one(assignment_doc)
    assignment = serialize_doc({"_id": result.inserted_id, **assignment_doc})
    return AssignmentResponse(**assignment)


@router.get("/patients/{patient_id}/sessions")
async def get_patient_sessions(
    patient_id: str,
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    pid = to_object_id(patient_id, "patient_id")

    await _assert_linked(db, doctor["id"], pid)

    cursor = db.sessions.find({"patient_id": pid}).sort("started_at", -1).limit(100)
    sessions = [serialize_doc(doc) async for doc in cursor]
    return {"sessions": sessions}


@router.get("/patients")
async def get_linked_patients(
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    links = db.doctor_patient_links.find({"doctor_id": ObjectId(doctor["id"]), "status": "active"})

    patients: list[dict] = []
    async for link in links:
        patient_doc = await db.users.find_one({"_id": link["patient_id"], "role": "patient"})
        if not patient_doc:
            continue
        patients.append(_public_patient(patient_doc))
    return {"patients": patients}


@router.get("/patients/search")
async def search_patients(
    q: str,
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    query = q.strip().lower()
    if len(query) < 2:
        return {"patients": []}

    links = db.doctor_patient_links.find({"doctor_id": ObjectId(doctor["id"]), "status": "active"})
    linked_ids: set[str] = set()
    async for link in links:
        linked_ids.add(str(link["patient_id"]))

    search_filter = {
        "role": "patient",
        "$or": [
            {"name": {"$regex": query, "$options": "i"}},
            {"email": {"$regex": query, "$options": "i"}},
            {"username": {"$regex": query, "$options": "i"}},
        ],
    }

    cursor = db.users.find(search_filter).sort("created_at", -1).limit(10)
    patients: list[dict] = []
    async for doc in cursor:
        serialized = _public_patient(doc)
        serialized["linked"] = serialized["id"] in linked_ids
        patients.append(serialized)

    return {"patients": patients}


@router.get("/patients/assignment-stats")
async def get_patient_assignment_stats(
    q: str | None = None,
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    links_cursor = db.doctor_patient_links.find({"doctor_id": ObjectId(doctor["id"]), "status": "active"})
    linked_ids: list[ObjectId] = []
    async for link in links_cursor:
        linked_ids.append(link["patient_id"])

    if not linked_ids:
        return {"stats": []}

    patient_filter: dict = {"_id": {"$in": linked_ids}, "role": "patient"}
    query = (q or "").strip()
    if query:
        patient_filter["$or"] = [
            {"name": {"$regex": query, "$options": "i"}},
            {"email": {"$regex": query, "$options": "i"}},
            {"username": {"$regex": query, "$options": "i"}},
        ]

    patient_docs = [doc async for doc in db.users.find(patient_filter)]
    if not patient_docs:
        return {"stats": []}

    patient_map: dict[str, dict] = {}
    patient_ids: list[ObjectId] = []
    for doc in patient_docs:
        public = _public_patient(doc)
        patient_map[public["id"]] = public
        patient_ids.append(doc["_id"])

    counts_map: dict[str, dict[str, int]] = {
        patient_id: {"assigned": 0, "in_progress": 0, "completed": 0, "total": 0}
        for patient_id in patient_map
    }

    counts_cursor = db.exercise_assignments.aggregate(
        [
            {
                "$match": {
                    "doctor_id": ObjectId(doctor["id"]),
                    "patient_id": {"$in": patient_ids},
                }
            },
            {
                "$group": {
                    "_id": {"patient_id": "$patient_id", "status": "$status"},
                    "count": {"$sum": 1},
                }
            },
        ]
    )

    async for row in counts_cursor:
        patient_id_str = str(row["_id"]["patient_id"])
        status = row["_id"].get("status")
        count = int(row.get("count", 0))
        if patient_id_str not in counts_map:
            continue
        counts_map[patient_id_str]["total"] += count
        if status in {"assigned", "in_progress", "completed"}:
            counts_map[patient_id_str][status] += count

    stats: list[PatientAssignmentStats] = []
    for patient_id, public_patient in patient_map.items():
        counts = counts_map[patient_id]
        stats.append(
            PatientAssignmentStats(
                patient=UserProfile(**public_patient),
                assigned_count=counts["assigned"],
                in_progress_count=counts["in_progress"],
                completed_count=counts["completed"],
                total_count=counts["total"],
            )
        )

    stats.sort(key=lambda item: item.patient.name.lower())
    return {"stats": [item.model_dump() for item in stats]}


@router.get("/patients/{patient_id}/report")
async def get_patient_report(
    patient_id: str,
    exercise_name: str | None = None,
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    pid = to_object_id(patient_id, "patient_id")
    await _assert_linked(db, doctor["id"], pid)

    session_filter: dict = {"patient_id": pid, "doctor_id": ObjectId(doctor["id"]), "status": "completed"}
    if exercise_name:
        session_filter["exercise_name"] = exercise_name

    sessions = [serialize_doc(doc) async for doc in db.sessions.find(session_filter).sort("started_at", 1)]
    scores = [float((session.get("summary") or {}).get("avg_final_score", 0.0)) for session in sessions]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0.0

    assignment_filter: dict = {"doctor_id": ObjectId(doctor["id"]), "patient_id": pid}
    if exercise_name:
        assignment_filter["exercise_name"] = exercise_name
    total_assignments = await db.exercise_assignments.count_documents(assignment_filter)
    completed_assignments = await db.exercise_assignments.count_documents({**assignment_filter, "status": "completed"})
    adherence = round((completed_assignments / total_assignments) * 100.0, 1) if total_assignments else 0.0

    progression_filter: dict = {"patient_id": pid}
    if exercise_name:
        progression_filter["exercise_name"] = exercise_name
    latest_snapshot = await db.progression_snapshots.find_one(progression_filter, sort=[("snapshot_at", -1)])

    return {
        "patient_id": patient_id,
        "exercise_name": exercise_name,
        "session_count": len(sessions),
        "avg_final_score": avg_score,
        "trend": _trend_label(scores[-5:]),
        "adherence_percent": adherence,
        "recent_scores": scores[-10:],
        "latest_progression": serialize_doc(latest_snapshot),
        "recent_sessions": sessions[-10:],
    }
