import time
import cv2
import numpy as np
from .param_server import shared_params

class GateStage:
    def __init__(self):
        self.state = "normal"
        self.start_time = 0.0
        self.barrier_clear_time = 0.0
        self.last_printed_state = ""
        
        self.is_barrier_down = False
        self.red_pixel_count = 0
        
        # 🔴 柵欄紅色的像素閾值 (Threshold)
        # 如果畫面中的紅色像素超過這個數字，就認定「柵欄放下來了」
        # (這個數字你可以根據實際鏡頭解析度去微調，通常 1000~3000 是一個好起點)
        self.red_threshold = 2000 

    def print_state(self):
        if self.state != self.last_printed_state:
            print(f"🛑 [Stage 6 Level Crossing] 目前進度: {self.state}")
            self.last_printed_state = self.state

    def process_yolo(self, yolo_detection):
        if not yolo_detection: return
        detection = str(yolo_detection).strip().lower()
        
        # 看到 STOP 標誌，進入接近柵欄的狀態
        if detection in ['stop', 'stop_sign'] and self.state == "normal":
            self.state = "approaching"
            print("🛑 看到 STOP 標誌！準備在柵欄前停車！")

    def process_vision(self, cv_image):
        p = shared_params.get("s6", {})
        self.red_threshold = p.get("red_threshold", 2000)
    
        lower_red_1 = np.array([p.get("red_h_low_1",0), p.get("red_s_low_1",70), p.get("red_v_low_1",50)])
        upper_red_1 = np.array([p.get("red_h_high_1",10), p.get("red_s_high_1",255), p.get("red_v_high_1",255)])
        lower_red_2 = np.array([p.get("red_h_low_2",170), p.get("red_s_low_2",70), p.get("red_v_low_2",50)])
        upper_red_2 = np.array([p.get("red_h_high_2",180), p.get("red_s_high_2",255), p.get("red_v_high_2",255)])
        
        """🌟 專門用來計算畫面中有多少「紅色」像素"""
        h, w = cv_image.shape[:2]
        
        # 轉換成 HSV 色彩空間
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        
        mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
        mask2 = cv2.inRange(hsv, lower_red_2, upper_red_2)
        
        # 合併兩個紅色遮罩
        red_mask = mask1 | mask2
        
        # ✂️ 【核心升級】：管狀視野！只看正中央！
        # 高度取 20%~80% (避開天花板與車頭)
        # 寬度只取 30%~70% (避開路邊的 STOP 標誌跟其他雜物)
        w_start = int(w * 0.3)
        w_end = int(w * 0.7)
        roi_mask = red_mask[int(h * 0.2) : int(h * 0.8), w_start : w_end]
        
        # 計算紅色像素的總數量
        self.red_pixel_count = cv2.countNonZero(roi_mask)

        if self.state == "approaching":
            print(f"🔍 [Debug] 眼前紅色像素: {self.red_pixel_count}")
        
        # 判斷柵欄是否放下
        self.is_barrier_down = self.red_pixel_count > self.red_threshold

    def get_action(self):
        p = shared_params.get("s6", {})
        debounce = p.get("debounce_duration", 2.0)
        self.print_state()

        if self.state == "approaching":
            # 繼續往前開，直到視線被紅白色柵欄塞滿！
            if self.is_barrier_down:
                print(f"🚧 柵欄已放下 (紅色像素: {self.red_pixel_count})！緊急煞車！")
                self.state = "waiting_at_barrier"
            return "line_follow", "straight"

        elif self.state == "waiting_at_barrier":
            # 死死踩住煞車！
            if not self.is_barrier_down:
                print("🚧 柵欄似乎升起了！開始 2 秒防彈跳確認...")
                self.state = "barrier_opening"
                self.barrier_clear_time = time.time()
            return "cmd_vel", (0.0, 0.0)

        elif self.state == "barrier_opening":
            # 你的 2 秒防彈跳邏輯 (Debounce Timer)
            if self.is_barrier_down:
                print("⚠️ 柵欄又掉下來了！退回等待狀態！")
                self.state = "waiting_at_barrier"
            elif time.time() - self.barrier_clear_time > debounce:
                print("✅ 確定柵欄已完全開啟！衝啊！")
                self.state = "done"
            return "cmd_vel", (0.0, 0.0) # 確認期間依然保持靜止

        elif self.state == "done":
            return "line_follow", "straight"

        return "line_follow", "straight"