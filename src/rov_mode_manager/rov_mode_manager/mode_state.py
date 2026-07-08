# rov_mode_manager/mode_state.py

from enum import Enum


class ROVMode(Enum):
    MANUAL = "manual"
    STABILIZE = "stabilize"
    DEPTH_HOLD = "depth_hold"
    AUTO_MISSION = "auto_mission"
