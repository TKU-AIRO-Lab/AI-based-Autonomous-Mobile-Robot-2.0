import time
import math
import cv2
import numpy as np
from .param_server import shared_params

try:
    from .slam_visualizer import get_nav_angle as _slam_get_angle
except ImportError:
    def _slam_get_angle(): return None

class TunnelStage:
    def __init__(self):
        self.state = "normal"
        self.start_time = 0.0
        self.last_printed_state = ""
        
        # 🌟 雷達與角度變數 (就是這裡漏掉宣告了！)
        self.front_dist = 999.0
        self.target_angle = 0.0     # 目前算出的最佳角度
        self.smoothed_angle = 0.0   # 避震器記憶體
        
        # 視覺變數
        self.is_red_line_detected = False
        self.red_threshold = 200
        
        self.is_vision_aligned = False
        self.lost_line_timer = 0.0
        self.seen_line_timer = 0.0 

    def print_state(self):
        if self.state != self.last_printed_state:
            print(f"🦇 [Stage 7 Tunnel] 目前狀態: {self.state}")
            self.last_printed_state = self.state

    def process_yolo(self, yolo_detection):
        if not yolo_detection: return
        detection = str(yolo_detection).strip().lower()
        
        if detection in ['tunnel', 'tunnel_sign'] and self.state == "normal":
            self.state = "entering"
            self.start_time = time.time()
            print("🦇 看到山洞標誌！保持尋線，直到黃白線完全消失...")

    def process_lidar(self, msg):
        """🌟 智慧路徑評分系統 (Local Cost Scoring) + 避震器"""
        front_scan = []
        max_score = -999.0
        best_angle = 0.0
        
        for i, dist in enumerate(msg.ranges):
            valid_dist = 3.5 if math.isinf(dist) or math.isnan(dist) else dist
            deg = math.degrees(msg.angle_min + i * msg.angle_increment)
            deg_norm = (deg + 180) % 360 - 180
            
            # 1. 紀錄正前方 (-20 到 20 度)，用來緊急防撞
            if abs(deg_norm) <= 20: 
                front_scan.append(valid_dist)
            
            # 2. 評分系統：只評估前方 180 度 (-90 到 90)
            if -90 <= deg_norm <= 90:
                # 基礎分數 = 距離 - 轉向懲罰
                penalty = abs(deg_norm) * 0.015 
                score = valid_dist - penalty
                
                # 🌟 慣性加分 (Consistency Bonus)：讓車子堅持選定的方向
                if abs(deg_norm - self.smoothed_angle) < 30:
                    score += 1.5 
                
                # 紀錄最高分的那條路徑
                if score > max_score:
                    max_score = score
                    best_angle = deg_norm
                    
        # 🌟 SLAM A* 全局路徑導向 + 低通濾波避震器
        # navigating 狀態下：把 SLAM 算出的出口角度混入本地評分
        # （SLAM 提供全局方向感，本地評分提供即時避障）
        if self.state == "navigating":
            slam_angle = _slam_get_angle()
            if slam_angle is not None:
                # 70% 本地評分（即時避障）+ 30% SLAM 全局方向（導向出口）
                best_angle = best_angle * 0.7 + slam_angle * 0.3

        self.smoothed_angle = 0.7 * self.smoothed_angle + 0.3 * best_angle
        self.front_dist = min(front_scan) if front_scan else 3.5
        self.target_angle = best_angle # 更新目標角度

    def process_vision(self, cv_image):
        if self.state not in ["back_on_track", "exiting"]: return
            
        h, w = cv_image.shape[:2]
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([180, 255, 255]))
        red_mask = mask1 | mask2
        
        roi_mask = red_mask[int(h * 0.5):h, int(w * 0.2):int(w * 0.8)]
        red_count = cv2.countNonZero(roi_mask)
        
        if self.state == "back_on_track" and red_count > 10:
            print(f"🔍 [Debug] 地板紅線像素: {red_count}")
        
        if red_count > self.red_threshold:
            self.is_red_line_detected = True

    def process_alignment(self, is_aligned):
        self.is_vision_aligned = is_aligned

    def get_action(self):
        p = shared_params.get("s7", {})
        blind_dur    = p.get("blind_forward_duration", 2.5)
        nav_kp       = p.get("nav_Kp", 0.012)
        emg_dist     = p.get("emergency_distance", 0.35)
        emg_steer    = p.get("emergency_steer", 0.8)
        red_thresh   = p.get("red_threshold", 200)
        exit_dur     = p.get("exit_duration", 3.0)
        lost_timeout = p.get("lost_line_timeout", 0.5)

        self.red_threshold = red_thresh
        
        self.print_state()
        elapsed = time.time() - self.start_time

        # 1. 尋線進場
        if self.state == "entering":
            if not self.is_vision_aligned:
                if self.lost_line_timer == 0.0: self.lost_line_timer = time.time()
                if time.time() - self.lost_line_timer > lost_timeout:
                    print("⬛ 進入黑暗！啟動 2.5 秒盲走...")
                    self.state = "blind_forward"
                    self.start_time = time.time()
            else:
                self.lost_line_timer = 0.0 
            return "line_follow", "straight"

        # 2. 盲走 2.5 秒
        elif self.state == "blind_forward":
            if elapsed > blind_dur:
                print("🦇 盲走結束！啟動【智慧評分路徑】導航！")
                self.state = "navigating"
                self.seen_line_timer = 0.0 
            return "cmd_vel", (0.12, 0.0) 

        # 3. 🦇 智慧路徑導航
        elif self.state == "navigating":
            if self.is_vision_aligned:
                if self.seen_line_timer == 0.0: self.seen_line_timer = time.time()
                if time.time() - self.seen_line_timer > lost_timeout:
                    print("☀️ 重見光明！看見黃白賽道，切換回尋線模式！")
                    self.state = "back_on_track"
                    self.start_time = time.time()
                    return "line_follow", "straight"
            else:
                self.seen_line_timer = 0.0 

            # 緊急防撞：如果正前方直接被柱子貼臉 (< 0.35m)
            if self.front_dist < emg_dist:
                # 順著已經決定的方向滑順地轉，閃過柱子
                steer = emg_steer if self.smoothed_angle > 0 else -0.8
                return "cmd_vel", (0.0, steer) 
            
            # ✅ 滑順導航
            else:
                Kp = nav_kp
                # 🌟 使用平滑過的角度來轉向
                steer = self.smoothed_angle * Kp
                steer = max(-0.5, min(0.5, steer)) 
                
                return "cmd_vel", (0.15, steer)

        # 4. 回歸賽道 (尋找紅線)
        elif self.state == "back_on_track":
            if self.is_red_line_detected:
                print("🚨 看到終點紅線了！準備最後衝刺！")
                self.state = "exiting"
                self.start_time = time.time()
                return "cmd_vel", (0.15, 0.0) 
            return "line_follow", "straight"

        # 5. 衝線
        elif self.state == "exiting":
            if elapsed > exit_dur:
                self.state = "done"
            return "cmd_vel", (0.15, 0.0)

        # 6. 完賽
        elif self.state == "done":
            return "stop", (0.0, 0.0)

        return "line_follow", "straight"