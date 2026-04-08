from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, status
from motor.motor_asyncio import AsyncIOMotorDatabase

from api_server.database import get_db
from api_server.deps import get_current_user
from api_server.schemas import LoginRequest, RefreshTokenRequest, RegisterRequest, TokenResponse, UserProfile
from api_server.security import (
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from api_server.utils import serialize_doc, utc_now

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse)
async def register(payload: RegisterRequest, db: AsyncIOMotorDatabase = Depends(get_db)) -> TokenResponse:
    email = payload.email.strip().lower()
    username = payload.username.strip().lower()

    existing = await db.users.find_one({"$or": [{"email": email}, {"username": username}]})
    if existing:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email or username already exists")

    user_doc = {
        "name": payload.name.strip(),
        "email": email,
        "username": username,
        "password_hash": hash_password(payload.password),
        "role": payload.role,
        "created_at": utc_now(),
    }
    result = await db.users.insert_one(user_doc)
    user_id = str(result.inserted_id)

    return TokenResponse(
        access_token=create_access_token(user_id=user_id, role=payload.role),
        refresh_token=create_refresh_token(user_id=user_id, role=payload.role),
    )


@router.post("/login", response_model=TokenResponse)
async def login(payload: LoginRequest, db: AsyncIOMotorDatabase = Depends(get_db)) -> TokenResponse:
    email = payload.email.strip().lower()
    user_doc = await db.users.find_one({"email": email})
    if not user_doc or not verify_password(payload.password, user_doc["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    user = serialize_doc(user_doc)
    return TokenResponse(
        access_token=create_access_token(user_id=user["id"], role=user["role"]),
        refresh_token=create_refresh_token(user_id=user["id"], role=user["role"]),
    )


@router.get("/me", response_model=UserProfile)
async def me(user: dict = Depends(get_current_user)) -> UserProfile:
    return UserProfile(**user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(payload: RefreshTokenRequest, db: AsyncIOMotorDatabase = Depends(get_db)) -> TokenResponse:
    try:
        token_payload = decode_token(payload.refresh_token)
    except TokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(exc)) from exc

    if token_payload.get("token_type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type")

    user_id = token_payload["sub"]
    if not ObjectId.is_valid(user_id):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject")
    user_doc = await db.users.find_one({"_id": ObjectId(user_id)})
    if not user_doc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    user = serialize_doc(user_doc)
    return TokenResponse(
        access_token=create_access_token(user_id=user["id"], role=user["role"]),
        refresh_token=create_refresh_token(user_id=user["id"], role=user["role"]),
    )
