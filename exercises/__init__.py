from .squats import Squats
from .sit_to_stand import SitToStand
from .marching import Marching
from .leg_raises import LegRaises
from .wall_pushups import WallPushups
from .hip_abduction import StandingHipAbduction
from .hip_extension import StandingHipExtension
from .heel_raises import HeelRaises
from .forward_arm_raises import ForwardArmRaises
from .side_arm_raises import SideArmRaises

EXERCISES = {
    "Squats": Squats(),
    "Sit-to-Stand": SitToStand(),
    "Marching": Marching(),
    "Leg Raises": LegRaises(),
    "Wall Push-ups": WallPushups(),
    "Standing Hip Abduction": StandingHipAbduction(),
    "Standing Hip Extension": StandingHipExtension(),
    "Heel Raises": HeelRaises(),
    "Forward Arm Raises": ForwardArmRaises(),
    "Side Arm Raises": SideArmRaises()
}
