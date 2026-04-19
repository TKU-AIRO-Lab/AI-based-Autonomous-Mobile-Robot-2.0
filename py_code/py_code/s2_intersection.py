import time
from .param_server import shared_params

class IntersectionStage:
    def __init__(self):
        self.state = "normal" 
        self.turn_direction = "straight"
        self.t_sign_seen_time = 0.0
        self.approach_duration = 7.5
        
        self.spin_start_time = 0.0
        self.spin_speed = 0.5
        
        # 🛡️ 【升級】：將盲轉時間拆分為左轉與右轉獨立控制！
        self.blind_spin_duration_left = 1.2   # 左轉 45 度需要的時間
        self.blind_spin_duration_right = 0.70  # 右轉 20 度需要的時間

    def process_yolo(self, yolo_detection, confirmed=False):
        if not yolo_detection:
            return

        detection = str(yolo_detection).strip().lower()

        # 必須 YOLO 確認後才開始茫走計時，避免一進 stage 2 就誤觸發
        if detection == 't' and self.state == "normal" and confirmed:
            self.state = "approaching"
            self.t_sign_seen_time = time.time()
            
        if self.state == "waiting":
            if detection == 'left':
                self.turn_direction = "left"
                self.state = "spinning"
                self.spin_start_time = time.time()
            elif detection == 'right':
                self.turn_direction = "right"
                self.state = "spinning"
                self.spin_start_time = time.time()

    def update_timer(self):
        if self.state == "approaching":
            if time.time() - self.t_sign_seen_time > self.approach_duration:
                self.state = "waiting"

    def process_vision(self, is_aligned):
        """處理動態視覺回饋"""
        if self.state == "spinning":
            elapsed_time = time.time() - self.spin_start_time
            
            # 🎯 根據當前轉彎方向，動態決定要閉著眼睛轉多久
            current_blind_duration = self.blind_spin_duration_left if self.turn_direction == "left" else self.blind_spin_duration_right
            
            # 只有在各自的盲轉時間結束後，才開始採納視覺對齊的結果！
            if elapsed_time > current_blind_duration:
                if is_aligned or (elapsed_time > 4.0):
                    self.state = "turning"

    def get_action(self):
        p = shared_params.get("s2", {})
        self.approach_duration = p.get("approach_duration", 7.5)
        self.spin_speed = p.get("spin_speed", 0.5)
        self.blind_spin_duration_left = p.get("blind_spin_duration_left", 1.57)
        self.blind_spin_duration_right = p.get("blind_spin_duration_right", 0.70)
        
        self.update_timer()
        
        if self.state == "waiting":
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
        
    def reset(self):
        self.state = "normal"
        self.turn_direction = "straight"
        self.t_sign_seen_time = 0.0
        self.spin_start_time = 0.0