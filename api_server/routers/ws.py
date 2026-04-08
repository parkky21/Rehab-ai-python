from bson import ObjectId
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from motor.motor_asyncio import AsyncIOMotorDatabase
from pydantic import ValidationError

from api_server.database import get_db
from api_server.runtime import RealtimeSessionRuntime
from api_server.schemas import RealtimeFrameMessage
from api_server.security import TokenError, decode_token
from api_server.utils import utc_now
from pipeline.progression import ProgressionState

router = APIRouter(tags=["realtime"])


async def _authenticate_websocket(token: str | None, db: AsyncIOMotorDatabase) -> dict | None:
    if not token:
        return None
    try:
        payload = decode_token(token)
    except TokenError:
        return None

    user_id = payload.get("sub")
    if not user_id or not ObjectId.is_valid(user_id):
        return None

    user = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user:
        return None

    return {
        "id": str(user["_id"]),
        "role": user["role"],
    }


@router.websocket("/ws/session")
async def session_websocket(websocket: WebSocket) -> None:
    db = get_db()
    token = websocket.query_params.get("token")
    assignment_id = websocket.query_params.get("assignment_id")

    await websocket.accept()

    if not assignment_id or not ObjectId.is_valid(assignment_id):
        await websocket.send_json({"type": "error", "detail": "Invalid assignment_id"})
        await websocket.close(code=1008)
        return

    user = await _authenticate_websocket(token, db)
    if not user:
        await websocket.send_json({"type": "error", "detail": "Invalid or expired token. Please sign in again."})
        await websocket.close(code=1008)
        return

    if user["role"] != "patient":
        await websocket.send_json({"type": "error", "detail": "Only patient accounts can open sessions."})
        await websocket.close(code=1008)
        return

    assignment = await db.exercise_assignments.find_one(
        {
            "_id": ObjectId(assignment_id),
            "patient_id": ObjectId(user["id"]),
            "status": {"$in": ["assigned", "in_progress"]},
        }
    )
    if not assignment:
        await websocket.send_json(
            {
                "type": "error",
                "detail": "Assignment not found or not active for this patient.",
            }
        )
        await websocket.close(code=1008)
        return

    await db.exercise_assignments.update_one(
        {"_id": assignment["_id"]},
        {"$set": {"status": "in_progress", "updated_at": utc_now()}},
    )

    runtime = RealtimeSessionRuntime(assignment["exercise_name"])
    session_doc = {
        "assignment_id": assignment["_id"],
        "doctor_id": assignment["doctor_id"],
        "patient_id": assignment["patient_id"],
        "exercise_name": assignment["exercise_name"],
        "target_reps": assignment["target_reps"],
        "status": "in_progress",
        "started_at": utc_now(),
    }
    session_result = await db.sessions.insert_one(session_doc)
    session_id = session_result.inserted_id

    await websocket.send_json(
        {
            "type": "session_started",
            "session_id": str(session_id),
            "exercise_name": assignment["exercise_name"],
            "target_reps": assignment["target_reps"],
        }
    )

    try:
        while True:
            message = await websocket.receive_json()
            event_type = message.get("type")

            if event_type == "session_end":
                break

            if event_type != "landmark_frame":
                await websocket.send_json({"type": "warning", "detail": "Unsupported event type"})
                continue

            try:
                frame = RealtimeFrameMessage(**message)
            except ValidationError as exc:
                await websocket.send_json({"type": "error", "detail": f"Invalid frame: {exc.errors()}"})
                continue

            try:
                result = runtime.process_frame([lm.model_dump() for lm in frame.landmarks])
            except ValueError as exc:
                await websocket.send_json({"type": "error", "detail": str(exc)})
                continue

            rep_event = result.get("rep_event")
            if rep_event is not None:
                await db.rep_events.insert_one(
                    {
                        "session_id": session_id,
                        "patient_id": assignment["patient_id"],
                        "doctor_id": assignment["doctor_id"],
                        "exercise_name": assignment["exercise_name"],
                        "rep_number": rep_event["rep_number"],
                        "scores": rep_event["scores"],
                        "rep_time": rep_event["rep_time"],
                        "rom_value": rep_event["rom_value"],
                        "created_at": utc_now(),
                    }
                )

            await websocket.send_json(
                {
                    "type": "frame_feedback",
                    "counter": result["counter"],
                    "stage": result["stage"],
                    "feedback": result["feedback"],
                    "feedback_rules": result["feedback_rules"],
                    "sway": result["sway"],
                    "rep_event": rep_event,
                }
            )
    except WebSocketDisconnect:
        pass
    finally:
        summary = runtime.finalize()
        await db.sessions.update_one(
            {"_id": session_id},
            {
                "$set": {
                    "status": "completed",
                    "ended_at": utc_now(),
                    "summary": summary,
                }
            },
        )
        await db.exercise_assignments.update_one(
            {"_id": assignment["_id"]},
            {"$set": {"status": "completed", "updated_at": utc_now()}},
        )

        score_docs = db.sessions.find(
            {
                "patient_id": assignment["patient_id"],
                "exercise_name": assignment["exercise_name"],
                "status": "completed",
            },
            {"summary.avg_final_score": 1},
        ).sort("started_at", -1).limit(5)

        recent_scores: list[float] = []
        async for score_doc in score_docs:
            recent_scores.append(float(score_doc.get("summary", {}).get("avg_final_score", 0.0)))

        progression = ProgressionState()
        progression.session_scores = list(reversed(recent_scores))
        progression_decision = progression.compute_progression()

        await db.progression_snapshots.insert_one(
            {
                "patient_id": assignment["patient_id"],
                "doctor_id": assignment["doctor_id"],
                "exercise_name": assignment["exercise_name"],
                "latest_score": float(summary.get("avg_final_score", 0.0)),
                "recent_scores": progression.session_scores,
                "target_reps": progression.target_reps,
                "target_rom_multiplier": progression.target_rom_multiplier,
                "sway_tolerance_multiplier": progression.sway_tolerance_multiplier,
                "decision": progression_decision,
                "snapshot_at": utc_now(),
            }
        )
