from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_server.config import get_settings
from api_server.database import mongo
from api_server.routers import auth, doctor, exercises, patient, ws


@asynccontextmanager
async def lifespan(_: FastAPI):
    await mongo.connect()
    yield
    await mongo.close()


settings = get_settings()
app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix=settings.api_prefix)
api_router.include_router(auth.router)
api_router.include_router(exercises.router)
api_router.include_router(doctor.router)
api_router.include_router(patient.router)
api_router.include_router(ws.router)


@api_router.get("/health")
async def health() -> dict:
    return {"status": "ok"}


app.include_router(api_router)
