from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from api_server.config import get_settings


class MongoManager:
    def __init__(self) -> None:
        self.client: AsyncIOMotorClient | None = None
        self.db: AsyncIOMotorDatabase | None = None

    async def connect(self) -> None:
        settings = get_settings()
        self.client = AsyncIOMotorClient(settings.mongodb_uri)
        self.db = self.client[settings.mongodb_db]
        await self._ensure_indexes()

    async def close(self) -> None:
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

    async def _ensure_indexes(self) -> None:
        if self.db is None:
            return

        await self.db.users.create_index("email", unique=True)
        await self.db.users.create_index("username", unique=True)

        await self.db.doctor_patient_links.create_index(
            [("doctor_id", 1), ("patient_id", 1)],
            unique=True,
        )

        await self.db.exercise_assignments.create_index([("patient_id", 1), ("status", 1)])
        await self.db.exercise_assignments.create_index([("doctor_id", 1), ("created_at", -1)])

        await self.db.sessions.create_index([("patient_id", 1), ("started_at", -1)])
        await self.db.sessions.create_index(
            [("doctor_id", 1), ("patient_id", 1), ("exercise_name", 1), ("started_at", -1)]
        )

        await self.db.rep_events.create_index([("session_id", 1), ("rep_number", 1)])

        await self.db.progression_snapshots.create_index(
            [("patient_id", 1), ("exercise_name", 1), ("snapshot_at", -1)]
        )


mongo = MongoManager()


def get_db() -> AsyncIOMotorDatabase:
    if mongo.db is None:
        raise RuntimeError("MongoDB connection is not initialized")
    return mongo.db
