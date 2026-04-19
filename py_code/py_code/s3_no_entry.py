import math
import time
from .param_server import shared_params

class NoEntryStage:
    def __init__(self):
        self.state = "normal"
        self.turn_direction = "straight"

        self.stop_duration = 0.5
        self.stop_start_time = 0.0

        self.spin_speed = 0.5
        self.spin_start_time = 0.0
        self.spin_degrees = 90.0

    def inherit_direction(self, previous_direction):
        self.turn_direction = previous_direction
        self.state = "stop"
        self.stop_start_time = time.time()

    def update_timer(self):
        if self.state == "stop":
            if time.time() - self.stop_start_time > self.stop_duration:
                self.state = "spinning"
                self.spin_start_time = time.time()

        elif self.state == "spinning":
            spin_duration = math.radians(self.spin_degrees) / max(self.spin_speed, 0.01)
            if time.time() - self.spin_start_time > spin_duration:
                self.state = "turning"

    def process_vision(self, is_aligned):
        """S3 改為固定時間控制，不使用視覺對齊判斷"""
        pass

    def get_action(self):
        p = shared_params.get("s3", {})
        self.spin_speed = p.get("spin_speed", 0.5)
        if self.turn_direction == "left":
            self.spin_degrees = p.get("blind_spin_duration_left", 90.0)
        else:
            self.spin_degrees = p.get("blind_spin_duration_right", 90.0)

        self.update_timer()

        if self.state == "stop":
            return "stop", 0.0
        elif self.state == "spinning":
            angular_z = self.spin_speed if self.turn_direction == "left" else -self.spin_speed
            return "spin", angular_z
        elif self.state == "turning":
            return "line_follow", self.turn_direction
        else:
            return "line_follow", "straight"

    def get_turn_direction(self):
        return self.turn_direction
