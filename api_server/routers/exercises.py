from fastapi import APIRouter

from api_server.exercise_factory import create_exercise_instance, list_exercises
from api_server.schemas import ExerciseInfo

router = APIRouter(prefix="/exercises", tags=["exercises"])


@router.get("", response_model=list[ExerciseInfo])
async def get_exercises() -> list[ExerciseInfo]:
    response: list[ExerciseInfo] = []
    for name in list_exercises():
        exercise = create_exercise_instance(name)
        response.append(
            ExerciseInfo(
                name=name,
                target_rom=exercise.config.target_rom,
                ideal_rep_time=exercise.config.ideal_rep_time,
                acceptable_sway=exercise.config.acceptable_sway,
            )
        )
    return response
