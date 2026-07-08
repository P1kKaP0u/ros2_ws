# rov_mode_manager/arm_state.py

from enum import Enum


class ArmState(Enum):
    DISARMED = 0
    ARMED = 1
