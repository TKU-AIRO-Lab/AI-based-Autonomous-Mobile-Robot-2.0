import time
from .param_server import shared_params

class NoEntryStage:
    def __init__(self):
        self.state = "normal" 
        self.turn_direction = "straight"
        
        self.approach_duration = 2.0 
        self.no_entry_seen_time = 0.0
        
        self.spin_speed = 0.5     
        self.spin_start_time = 0.0
        
        # 🛡️ 【升級】：將盲轉時間拆分為左轉與右轉獨立控制！
        self.blind_spin_duration_left = 1.57   # 左轉 45 度需要的時間
        self.blind_spin_duration_right = 0.70  # 右轉 20 度需要的時間 

    def inherit_direction(self, previous_direction):
        self.turn_direction = previous_direction
        self.state = "approaching"
        self.no_entry_seen_time = time.time()

    def update_timer(self):
        if self.state == "approaching":
            if time.time() - self.no_entry_seen_time > self.approach_duration:
                self.state = "spinning"
                self.spin_start_time = time.time()

    def process_vision(self, is_aligned):
        """處理動態視覺回饋"""
        if self.state == "spinning":
            elapsed_time = time.time() - self.spin_start_time
            
            # 🎯 根據轉向記憶，動態決定盲轉時間
            current_blind_duration = self.blind_spin_duration_left if self.turn_direction == "left" else self.blind_spin_duration_right
            
            if elapsed_time > current_blind_duration:
                if is_aligned or (elapsed_time > 4.0):
                    self.state = "turning"

    def get_action(self):
        p = shared_params.get("s3", {})
        self.approach_duration = p.get("approach_duration", 2.0)
        self.spin_speed = p.get("spin_speed", 0.5)
        self.blind_spin_duration_left = p.get("blind_spin_duration_left", 1.57)
        self.blind_spin_duration_right = p.get("blind_spin_duration_right", 0.70)
        
        self.update_timer()
        
        if self.state == "approaching":
            return "line_follow", "straight"
        elif self.state == "spinning":
            angular_z = self.spin_speed if self.turn_direction == "left" else -self.spin_speed
            return "spin", angular_z
        elif self.state == "turning":
            return "line_follow", "straight"
        else:
            return "line_follow", "straight"
            
    def get_turn_direction(self):
        return self.turn_direction