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
    patients = [
        {"name": "Aarav Pathak", "email": "pathak@rehabai.local", "username": "pathak"},
        {"name": "Neha Kapoor", "email": "neha.k@rehabai.local", "username": "neha_k"},
        {"name": "Rohan Mehta", "email": "rohan.m@rehabai.local", "username": "rohan_m"},
        {"name": "Priya Sharma", "email": "priya.s@rehabai.local", "username": "priya_s"},
        {"name": "Vikram Rao", "email": "vikram.r@rehabai.local", "username": "vikram_r"},
        {"name": "Ananya Iyer", "email": "ananya.i@rehabai.local", "username": "ananya_i"},
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
    for idx, patient in enumerate(patients, start=1):
        patient_doc = await db.users.find_one(
            {
                "role": "patient",
                "$or": [
                    {"email": patient["email"]},
                    {"username": patient["username"]},
                ],
            }
        )
        patient_hash = hash_password("12345678")

        if not patient_doc:
            result = await db.users.insert_one(
                {
                    "name": patient["name"],
                    "email": patient["email"],
                    "username": patient["username"],
                    "password_hash": patient_hash,
                    "role": "patient",
                    "created_at": utc_now(),
                }
            )
            patient_id = result.inserted_id
        else:
            patient_id = _doc_id(patient_doc)
            await db.users.update_one(
                {"_id": patient_id},
                {
                    "$set": {
                        "name": patient["name"],
                        "username": patient["username"],
                        "password_hash": patient_hash,
                    }
                },
            )

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

        assignment_templates = [
            {"exercise_name": "Squats", "target_reps": 10, "status": "assigned"},
            {"exercise_name": "Heel Raises", "target_reps": 12, "status": "in_progress"},
            {"exercise_name": "Marching", "target_reps": 14, "status": "completed"},
        ]

        for template in assignment_templates:
            await db.exercise_assignments.update_one(
                {
                    "doctor_id": doctor_id,
                    "patient_id": patient_id,
                    "exercise_name": template["exercise_name"],
                    "status": template["status"],
                },
                {
                    "$setOnInsert": {
                        "doctor_id": doctor_id,
                        "patient_id": patient_id,
                        "exercise_name": template["exercise_name"],
                        "target_reps": template["target_reps"],
                        "status": template["status"],
                        "notes": "Demo assignment",
                        "created_at": utc_now(),
                        "updated_at": utc_now(),
                    }
                },
                upsert=True,
            )

    print("Seed complete")
    print("Doctor login: doctor.demo@rehabai.local / Doctor@123")
    print("All seeded patient passwords: 12345678")

    await mongo.close()


if __name__ == "__main__":
    asyncio.run(seed())
