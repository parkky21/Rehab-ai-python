from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api_server.database import get_db
from api_server.deps import require_role
from api_server.exercise_factory import list_exercises
from api_server.schemas import AssignmentCreateRequest, AssignmentResponse, PatientLinkRequest
from api_server.utils import serialize_doc, to_object_id, utc_now

router = APIRouter(prefix="/doctor", tags=["doctor"])


@router.post("/patients/link")
async def link_patient(
    payload: PatientLinkRequest,
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> dict:
    patient_id = to_object_id(payload.patient_id, "patient_id")
    patient_doc = await db.users.find_one({"_id": patient_id, "role": "patient"})
    if not patient_doc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Patient not found")

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

    return {"status": "linked", "patient_id": payload.patient_id}


@router.post("/assignments", response_model=AssignmentResponse)
async def create_assignment(
    payload: AssignmentCreateRequest,
    doctor: dict = Depends(require_role({"doctor"})),
    db: AsyncIOMotorDatabase = Depends(get_db),
) -> AssignmentResponse:
    if payload.exercise_name not in list_exercises():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported exercise")

    patient_id = to_object_id(payload.patient_id, "patient_id")

    linked = await db.doctor_patient_links.find_one(
        {
            "doctor_id": ObjectId(doctor["id"]),
            "patient_id": patient_id,
            "status": "active",
        }
    )
    if not linked:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Patient is not linked to this doctor")

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

    linked = await db.doctor_patient_links.find_one(
        {
            "doctor_id": ObjectId(doctor["id"]),
            "patient_id": pid,
            "status": "active",
        }
    )
    if not linked:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Patient is not linked to this doctor")

    cursor = db.sessions.find({"patient_id": pid}).sort("started_at", -1).limit(100)
    sessions = [serialize_doc(doc) async for doc in cursor]
    return {"sessions": sessions}
