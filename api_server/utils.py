from datetime import UTC, datetime

from bson import ObjectId
from fastapi import HTTPException, status


def utc_now() -> datetime:
    return datetime.now(UTC)


def to_object_id(value: str, field_name: str = "id") -> ObjectId:
    if not ObjectId.is_valid(value):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid ObjectId for {field_name}",
        )
    return ObjectId(value)


def serialize_doc(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    data = dict(doc)
    if "_id" in data:
        data["id"] = str(data.pop("_id"))
    for key, value in list(data.items()):
        if isinstance(value, ObjectId):
            data[key] = str(value)
    return data
