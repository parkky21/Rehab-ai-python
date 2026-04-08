from exercises.forward_arm_raises import ForwardArmRaises
from exercises.heel_raises import HeelRaises
from exercises.hip_abduction import StandingHipAbduction
from exercises.hip_extension import StandingHipExtension
from exercises.leg_raises import LegRaises
from exercises.marching import Marching
from exercises.side_arm_raises import SideArmRaises
from exercises.sit_to_stand import SitToStand
from exercises.squats import Squats
from exercises.wall_pushups import WallPushups


EXERCISE_CLASS_MAP = {
    "Squats": Squats,
    "Sit-to-Stand": SitToStand,
    "Marching": Marching,
    "Leg Raises": LegRaises,
    "Wall Push-ups": WallPushups,
    "Standing Hip Abduction": StandingHipAbduction,
    "Standing Hip Extension": StandingHipExtension,
    "Heel Raises": HeelRaises,
    "Forward Arm Raises": ForwardArmRaises,
    "Side Arm Raises": SideArmRaises,
}


def list_exercises() -> list[str]:
    return list(EXERCISE_CLASS_MAP.keys())


def create_exercise_instance(exercise_name: str):
    cls = EXERCISE_CLASS_MAP.get(exercise_name)
    if cls is None:
        raise ValueError(f"Unsupported exercise: {exercise_name}")
    return cls()
