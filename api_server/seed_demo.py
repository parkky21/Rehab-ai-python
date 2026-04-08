import asyncio

from bson import ObjectId

from api_server.database import mongo
from api_server.security import hash_password
from api_server.utils import utc_now


def _doc_id(doc: dict | None) -> ObjectId | None:
    return doc.get("_id") if doc else None


async def seed() -> None:
    await mongo.connect()
    db = mongo.db
    if db is None:
        raise RuntimeError("MongoDB not connected")

    doctor_email = "doctor.demo@rehabai.local"
    patient_emails = [
        "patient.one@rehabai.local",
        "patient.two@rehabai.local",
    ]

    doctor_doc = await db.users.find_one({"email": doctor_email})
    if not doctor_doc:
        result = await db.users.insert_one(
            {
                "name": "Dr. Demo",
                "email": doctor_email,
                "username": "doctor_demo",
                "password_hash": hash_password("Doctor@123"),
                "role": "doctor",
                "created_at": utc_now(),
            }
        )
        doctor_id = result.inserted_id
    else:
        doctor_id = _doc_id(doctor_doc)

    if doctor_id is None:
        raise RuntimeError("Doctor seed failed")

    patient_ids: list[ObjectId] = []
    for idx, email in enumerate(patient_emails, start=1):
        patient_doc = await db.users.find_one({"email": email})
        if not patient_doc:
            result = await db.users.insert_one(
                {
                    "name": f"Patient {idx}",
                    "email": email,
                    "username": f"patient_{idx}",
                    "password_hash": hash_password("Patient@123"),
                    "role": "patient",
                    "created_at": utc_now(),
                }
            )
            patient_id = result.inserted_id
        else:
            patient_id = _doc_id(patient_doc)

        if patient_id is None:
            continue

        patient_ids.append(patient_id)
        await db.doctor_patient_links.update_one(
            {"doctor_id": doctor_id, "patient_id": patient_id},
            {
                "$setOnInsert": {
                    "doctor_id": doctor_id,
                    "patient_id": patient_id,
                    "status": "active",
                    "created_at": utc_now(),
                }
            },
            upsert=True,
        )

        await db.exercise_assignments.update_one(
            {
                "doctor_id": doctor_id,
                "patient_id": patient_id,
                "exercise_name": "Squats",
                "status": {"$in": ["assigned", "in_progress"]},
            },
            {
                "$setOnInsert": {
                    "doctor_id": doctor_id,
                    "patient_id": patient_id,
                    "exercise_name": "Squats",
                    "target_reps": 10,
                    "status": "assigned",
                    "notes": "Demo assignment",
                    "created_at": utc_now(),
                }
            },
            upsert=True,
        )

    print("Seed complete")
    print("Doctor login: doctor.demo@rehabai.local / Doctor@123")
    print("Patient login: patient.one@rehabai.local / Patient@123")
    print("Patient login: patient.two@rehabai.local / Patient@123")

    await mongo.close()


if __name__ == "__main__":
    asyncio.run(seed())
