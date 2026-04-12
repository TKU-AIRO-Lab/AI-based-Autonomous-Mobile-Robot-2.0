import time
from .param_server import shared_params

class ObstacleStage:
    def __init__(self):
        self.state = "normal" 
        self.obstacle_seen_time = 0.0
        
        # ⏱️ 避障模式要維持幾秒？(根據你賽道障礙物的長度來微調)
        self.avoid_duration = 5.0 

    def process_yolo(self, yolo_detection):
        if not yolo_detection:
            return

        detection = str(yolo_detection).strip().lower()

        # 看到施工標誌，進入避障模式
        if detection in ['construction', 'obstacle'] and self.state == "normal":
            self.state = "avoiding"
            self.obstacle_seen_time = time.time()

    def update_timer(self):
        if self.state == "avoiding":
            if time.time() - self.obstacle_seen_time > self.avoid_duration:
                self.state = "passed" # 繞過障礙物了，恢復正常！

    def get_action(self):
        p = shared_params.get("s4", {})
        self.avoid_duration = p.get("avoid_duration", 5.0)
        self.update_timer()
        
        if self.state == "avoiding":
            # 告訴小腦：縮小遮罩，死盯右邊的白線！
            return "line_follow", "follow_right"
        else:
            return "line_follow", "straight"