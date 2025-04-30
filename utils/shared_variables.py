from typing import Optional

class SharedVariables:
    """
    A container class to store shared variables used across the facial tracking and lip sync pipeline.

    Attributes:
        ear_left (Optional[float]): Left Eye Aspect Ratio (EAR) value.
        ear_right (Optional[float]): Right Eye Aspect Ratio (EAR) value.
        mar (Optional[float]): Mouth Aspect Ratio (MAR) value.
        ebr_left (Optional[float]): Left Eye Blink Ratio (or similar metric).
        ebr_right (Optional[float]): Right Eye Blink Ratio (or similar metric).
        lip_sync_value (Optional[float]): Value indicating lip sync activity.
        yaw (Optional[float]): Head pose yaw angle.
        pitch (Optional[float]): Head pose pitch angle.
        roll (Optional[float]): Head pose roll angle.
    """
    def __init__(self) -> None:
        self.ear_left: Optional[float] = None
        self.ear_right: Optional[float] = None
        self.mar: Optional[float] = None
        self.ebr_left: Optional[float] = None
        self.ebr_right: Optional[float] = None
        self.lip_sync_value: Optional[float] = None
        self.yaw: Optional[float] = None
        self.pitch: Optional[float] = None
        self.roll: Optional[float] = None

    def reset(self) -> None:
        """
        Reset all shared variables to their initial state (None).
        Useful when restarting a session or recovering from a tracking failure.
        """
        self.ear_left = None
        self.ear_right = None
        self.mar = None
        self.ebr_left = None
        self.ebr_right = None
        self.lip_sync_value = None
        self.yaw = None
        self.pitch = None
        self.roll = None