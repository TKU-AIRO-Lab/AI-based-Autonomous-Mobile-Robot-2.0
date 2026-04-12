import time
import math
from .param_server import shared_params

class ParkingStage:
    def __init__(self):
        self.state = "normal"
        self.start_time = 0.0
        self.last_printed_state = ""
        self.left_dist = 999.0
        self.right_dist = 999.0
        self.chosen_dir = None
        self.is_vision_aligned = False
        self.lost_line_timer = 0.0
        
        # 🌟 新增：記憶開關！用來記住「我已經看過離開標誌了」
        self.exit_sign_seen = False 

    def print_state(self):
        if self.state != self.last_printed_state:
            print(f"🚗 [Stage 5 Parking] {self.state}")
            self.last_printed_state = self.state

    def process_yolo(self, yolo_detection):
        if not yolo_detection: return
        detection = str(yolo_detection).strip().lower()
        
        if detection == 'parking' and self.state == "normal":
            self.state = "forward_delay"
            self.start_time = time.time()
            
        elif 'left' in detection and self.state in ["parked_2s", "drive_out_2s", "spin_out_90", "exit_cruising"]:
            # 🌟 核心修改：看到標誌不馬上轉！只把記憶開關打開！
            if not self.exit_sign_seen:
                self.exit_sign_seen = True
                print("🏁 [記憶寫入] 看到左轉標誌！將在雙黃線結束後執行轉彎！")

    def process_lidar(self, msg):
        left_scan = []
        right_scan = []
        for i, dist in enumerate(msg.ranges):
            if dist < 0.01 or math.isinf(dist) or math.isnan(dist): continue
            deg = (math.degrees(msg.angle_min + i * msg.angle_increment)) % 360
            if 70 <= deg <= 110: left_scan.append(dist)
            elif 250 <= deg <= 290: right_scan.append(dist)
        self.left_dist = min(left_scan) if left_scan else 999.0
        self.right_dist = min(right_scan) if right_scan else 999.0

    def process_vision(self, is_aligned):
        self.is_vision_aligned = is_aligned

    def get_action(self):
        p = shared_params.get("s5", {})
        self.print_state()
        elapsed = time.time() - self.start_time

        if self.state == "forward_delay":
            if elapsed > p.get("forward_delay", 5.0):
                self.state = "turn_in_90"
                self.start_time = time.time()
            return "line_follow", "straight"

        elif self.state == "turn_in_90":
            if elapsed > p.get("turn_in_duration", 1.57):
                self.state = "double_yellow_cruising"
                self.start_time = time.time()
                self.lost_line_timer = 0.0
            return "cmd_vel", (0.0, 1.0) 

        elif self.state == "double_yellow_cruising":
            if not self.is_vision_aligned:
                if self.lost_line_timer == 0.0: self.lost_line_timer = time.time()
                if time.time() - self.lost_line_timer > 0.5:
                    self.state = "blind_forward_2s"
                    self.start_time = time.time()
            else:
                self.lost_line_timer = 0.0
            return "line_follow", "double_yellow"

        elif self.state == "blind_forward_2s":
            if elapsed > p.get("blind_forward_duration", 2.0):
                self.state = "radar_scan"
                self.start_time = time.time()
            return "cmd_vel", (0.12, 0.0)

        elif self.state == "radar_scan":
            if elapsed > 0.5:
                if self.left_dist >  p.get("empty_threshold", 0.45):
                    self.chosen_dir = "left"
                    self.state = "spin_into_spot_90"
                    self.start_time = time.time()
                elif self.right_dist > p.get("empty_threshold", 0.45):
                    self.chosen_dir = "right"
                    self.state = "spin_into_spot_90"
                    self.start_time = time.time()
                else:
                    self.state = "blind_forward_2s"
                    self.start_time = time.time()
            return "cmd_vel", (0.0, 0.0)

        elif self.state == "spin_into_spot_90":
            if elapsed > 1.57:
                self.state = "drive_in_2s"
                self.start_time = time.time()
            angular_z = 1.0 if self.chosen_dir == "left" else -1.0
            return "cmd_vel", (0.0, angular_z)

        elif self.state == "drive_in_2s":
            if elapsed > p.get("drive_in_duration", 2.0):
                self.state = "parked_2s"
                self.start_time = time.time()
            return "cmd_vel", (0.12, 0.0)

        elif self.state == "parked_2s":
            if elapsed > p.get("parked_duration", 2.0):
                self.state = "drive_out_2s"
                self.start_time = time.time()
            return "cmd_vel", (0.0, 0.0)

        elif self.state == "drive_out_2s":
            if elapsed > p.get("drive_out_duration", 2.0):
                self.state = "spin_out_90"
                self.start_time = time.time()
            return "cmd_vel", (-0.12, 0.0)

        elif self.state == "spin_out_90":
            if elapsed > p.get("spin_out_duration", 1.57):
                self.state = "exit_cruising"
                self.lost_line_timer = 0.0 # 進入巡航前重置計時器
            angular_z = 1.0 if self.chosen_dir == "left" else -1.0
            return "cmd_vel", (0.0, angular_z)

        elif self.state == "exit_cruising":
            # 🌟 核心修改：記憶延遲執行邏輯
            if self.exit_sign_seen:
                if not self.is_vision_aligned:
                    # 如果看到標誌，而且雙黃線斷掉了！
                    if self.lost_line_timer == 0.0: self.lost_line_timer = time.time()
                    
                    # 確定失去雙黃線超過 0.5 秒 (代表真正走到了路口邊緣)
                    if time.time() - self.lost_line_timer > p.get("lost_line_timeout", 0.5):
                        print("🛣️ 雙黃線結束！開始執行左轉出站！")
                        self.state = "final_turn_out"
                        self.start_time = time.time()
                else:
                    # 如果還看得到雙黃線，就重置計時器，繼續乖乖往前開
                    self.lost_line_timer = 0.0
            
            return "line_follow", "double_yellow"
            
        elif self.state == "final_turn_out":
            if elapsed > p.get("final_turn_duration", 1.57):
                self.state = "done"
            return "cmd_vel", (0.0, 1.0)
            
        elif self.state == "done":
            return "line_follow", "straight"

        return "line_follow", "straight"