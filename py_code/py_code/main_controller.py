import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String, Int32
from cv_bridge import CvBridge
from py_code.param_server import shared_params
import cv2
import numpy as np
import json
import math
import os
import time
from collections import deque

# ── 相機校正 ──────────────────────────────────────────────────────────────────
_CALIB_PATH = os.path.expanduser('~/ros2_ws/calibration.npz')

def _load_calibration():
    if not os.path.exists(_CALIB_PATH):
        return None, None
    data = np.load(_CALIB_PATH)
    return data['mtx'], data['dist']

_CAM_MTX, _CAM_DIST = _load_calibration()
_CAM_MAPS = {}   # 快取 remap 查找表，key = (height, width)

def undistort(frame):
    """套用相機校正去畸變。若校正檔不存在則直接回傳原圖。"""
    if _CAM_MTX is None:
        return frame
    h, w = frame.shape[:2]
    if (h, w) not in _CAM_MAPS:
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(_CAM_MTX, _CAM_DIST, (w, h), 0, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(
            _CAM_MTX, _CAM_DIST, None, new_mtx, (w, h), cv2.CV_16SC2
        )
        _CAM_MAPS[(h, w)] = (map1, map2)
    map1, map2 = _CAM_MAPS[(h, w)]
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
# ─────────────────────────────────────────────────────────────────────────────

YOLO_WINDOW_SIZE   = 8   # 滑動窗口大小（看最近幾幀）
YOLO_CONFIRM_THRESH = 5  # 窗口內出現幾次同一 label 就算確認

WHEEL_DELAY = 1.2  # 鏡頭超前輪軸的補償延遲（秒）

# 匯入你的關卡模組
from py_code.s1_traffic_light import TrafficLightStage
from py_code.s2_intersection import IntersectionStage
from py_code.s3_no_entry import NoEntryStage
from py_code.s4_obstacle import ObstacleStage
from py_code.s5_parking import ParkingStage
from py_code.s6_gate import GateStage
from py_code.s7_tunnel import TunnelStage
from py_code.line import LineFollowerBase

class MainController(Node):
    def __init__(self):
        super().__init__('main_controller')
        self.bridge = CvBridge()
        
        self.cs = 1 # cs = current stage
        self.is_running = False
        
        # s = stage
        self.s1 = TrafficLightStage() 
        self.s2 = IntersectionStage()
        self.s3 = NoEntryStage()
        self.s4 = ObstacleStage()
        self.s5 = ParkingStage() 
        self.s6 = GateStage()
        self.s7 = TunnelStage()
        self.line_follower = LineFollowerBase()
        
        self.yolo_sub = self.create_subscription(String, '/yolo', self.yolo_callback, 10)
        self.image_sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.params_sub = self.create_subscription(String, '/tuning_params', self.params_callback, 10)
        self.line_view_pub = self.create_publisher(Image, '/line_view', 10)
        # 隨時切換關卡
        self.stage_sub = self.create_subscription(Int32, '/set_stage', self.set_stage_callback, 10)
        
        self.lidar_override = False
        self.lidar_steer = 0.0

        # 鏡頭超前輪軸補償：緩衝循線 angular.z，延遲 WHEEL_DELAY 秒後才套用
        self._steer_buffer: deque = deque()

        # YOLO 防呆：滑動窗口，最近 YOLO_WINDOW_SIZE 幀裡出現 YOLO_CONFIRM_THRESH 次就確認
        self._yolo_window: deque = deque(maxlen=YOLO_WINDOW_SIZE)

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info("現在進入 Stage 1: 等待綠燈")

    def _publish_with_delay(self, twist_cmd):
        """循線專用 publish：angular.z 延遲 WHEEL_DELAY 秒，補償鏡頭超前輪軸的偏移。"""
        now = time.time()
        self._steer_buffer.append((now, twist_cmd.angular.z))

        target_t = now - WHEEL_DELAY
        while len(self._steer_buffer) > 1 and self._steer_buffer[1][0] <= target_t:
            self._steer_buffer.popleft()

        delayed_cmd = Twist()
        delayed_cmd.linear.x = twist_cmd.linear.x
        delayed_cmd.angular.z = self._steer_buffer[0][1]
        self.cmd_vel_pub.publish(delayed_cmd)

    def _yolo_confirmed(self, label: str) -> bool:
        """回傳 True 表示最近 YOLO_WINDOW_SIZE 幀裡，label 出現次數達到 YOLO_CONFIRM_THRESH"""
        self._yolo_window.append(label)
        count = sum(1 for d in self._yolo_window if d == label)
        confirmed = count >= YOLO_CONFIRM_THRESH
        if confirmed:
            self._yolo_window.clear()  # 確認後清空，避免重複觸發
        return confirmed

    def params_callback(self, msg):
        try:
            new_params = json.loads(msg.data)
            shared_params.update(new_params)
        except Exception:
            pass

    def set_stage_callback(self, msg):
        """可以強制切換關卡並直接發車"""
        self.cs = msg.data
        self.is_running = True
        self.get_logger().info(f"🚀 [DEBUG] 強制跳到Stage {self.cs}!")

    def yolo_callback(self, msg):
        detection = msg.data.lower()
        confirmed = self._yolo_confirmed(detection)
        if confirmed:
            self.get_logger().info(f"[YOLO 確認] {detection} ({YOLO_CONFIRM_THRESH}/{YOLO_WINDOW_SIZE})")

        if self.cs == 1:
            if confirmed and self.s1.check_start_condition(detection):
                self.is_running = True
                self.cs = 2
                self.get_logger().info("現在進入 Stage 2 (綠燈確認)")
                go_cmd = Twist()
                go_cmd.linear.x = 0.10
                self.cmd_vel_pub.publish(go_cmd)
            else:
                self.cmd_vel_pub.publish(self.s1.get_stop_twist())

        elif self.cs == 2:
            if confirmed and detection == 't':
                self.get_logger().info("Stage 2: T 路口號誌確認，開始茫走")
            self.s2.process_yolo(detection, confirmed)
            if confirmed and detection == 'no_entry':
                memory_direction = self.s2.get_turn_direction()
                self.s3.inherit_direction(memory_direction)
                self.cs = 3
                self.get_logger().info("現在進入 Stage 3 (禁止進入確認)")

        elif self.cs == 3:
            if confirmed and detection == 'obstacle':
                self.cs = 4
                self.s4.process_yolo(detection)
                self.get_logger().info("現在進入 Stage 4 (障礙物確認)")

        elif self.cs == 4:
            if confirmed and detection == 'parking':
                self.cs = 5
                self.s5.process_yolo(detection)
                self.get_logger().info("現在進入 Stage 5 (停車號誌確認)")

        elif self.cs == 5:
            self.s5.process_yolo(detection)
            # s5 完成後繼續尋線，直到確認看到 stop 才進入 Stage 6
            if self.s5.state == "done" and confirmed and detection == 'stop':
                self.cs = 6
                self.get_logger().info("現在進入 Stage 6 (stop確認)")

        if self.cs == 6:
            if confirmed and detection == 'stop':
                self.s6.process_yolo(detection)
            # s6 完成後繼續尋線，直到確認看到 tunnel 才進入 Stage 7
            if self.s6.state == "done" and confirmed and detection == 'tunnel':
                self.cs = 7
                self.get_logger().info("現在進入 Stage 7 (tunnel確認)")

        if self.cs == 7:
            if confirmed and detection == 'tunnel':
                self.s7.process_yolo(detection)

    def scan_callback(self, msg):
        # 餵雷達資料給第五關找車位
        if self.cs == 5:
            self.s5.process_lidar(msg)

        # 第四關的泛用型雷達避障
        if self.cs != 4:
            self.lidar_override = False
            return
            
        front_scan = []
        left_scan  = []
        right_scan = []
        for i, dist in enumerate(msg.ranges):
            if dist < 0.01 or math.isinf(dist) or math.isnan(dist): continue
            deg = (math.degrees(msg.angle_min + i * msg.angle_increment)) % 360
            if deg <= 20 or deg >= 340:   # ±20° 正前方
                front_scan.append(dist)
            elif 20 < deg <= 60:          # 前左
                left_scan.append(dist)
            elif 300 <= deg < 340:        # 前右
                right_scan.append(dist)

        ffd = min(front_scan) if front_scan else float('inf')  # front
        lfd = min(left_scan)  if left_scan  else float('inf')  # left front
        rfd = min(right_scan) if right_scan else float('inf')  # right front

        p4 = shared_params.get("s4", {})
        danger_zone = p4.get("danger_zone", 0.35)
        avoid_steer = p4.get("avoid_steer", 0.8)

        if ffd < danger_zone:
            # 正前方有障礙 → 往空間較大的那側閃
            self.lidar_override = True
            self.lidar_steer = avoid_steer if lfd > rfd else -avoid_steer
        elif lfd < danger_zone:
            self.lidar_override = True
            self.lidar_steer = -avoid_steer   # 左側有障礙 → 往右閃
        elif rfd < danger_zone:
            self.lidar_override = True
            self.lidar_steer = avoid_steer    # 右側有障礙 → 往左閃
        else:
            self.lidar_override = False

    def image_callback(self, msg):
        try: cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e: return
        cv_image = undistort(cv_image)

        final_view = None

        if not self.is_running:
            try:
                _, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="straight")
            except Exception:
                pass
            if final_view is not None:
                try:
                    img_msg = Image()
                    img_msg.header.stamp = self.get_clock().now().to_msg()
                    img_msg.height = final_view.shape[0]
                    img_msg.width = final_view.shape[1]
                    img_msg.encoding = 'bgr8'
                    img_msg.is_bigendian = 0
                    img_msg.step = final_view.shape[1] * 3
                    img_msg.data = final_view.tobytes()
                    self.line_view_pub.publish(img_msg)
                except Exception:
                    pass
            return

        if self.cs == 2:
            action, value = self.s2.get_action()
            if action == "stop":
                stop_cmd = Twist()
                self.cmd_vel_pub.publish(stop_cmd)
                _, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="straight")
            elif action == "spin":
                spin_dir = "spin_left" if value > 0 else "spin_right"
                spin_cmd = Twist()
                spin_cmd.angular.z = float(value)
                self.cmd_vel_pub.publish(spin_cmd)
                _, final_view, is_aligned = self.line_follower.process_image(cv_image, turn_direction=spin_dir)
                self.s2.process_vision(is_aligned) 
            elif action == "line_follow":
                twist_cmd, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="straight")
                self._publish_with_delay(twist_cmd)

        elif self.cs == 3:
            action, value = self.s3.get_action()
            if action == "spin":
                spin_dir = "spin_left" if value > 0 else "spin_right"
                spin_cmd = Twist()
                spin_cmd.angular.z = float(value)
                self.cmd_vel_pub.publish(spin_cmd)
                _, final_view, is_aligned = self.line_follower.process_image(cv_image, turn_direction=spin_dir)
                self.s3.process_vision(is_aligned) 
            elif action == "line_follow":
                twist_cmd, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="straight")
                self._publish_with_delay(twist_cmd)

        elif self.cs == 4:
            # 讓 s4 決定尋線模式（正常=nearsighted，看到障礙=follow_right）
            action, value = self.s4.get_action()
            turn_dir = value if action == "line_follow" else "nearsighted"

            if self.lidar_override:
                # 雷達緊急閃避，蓋過一切
                twist_cmd = Twist()
                twist_cmd.linear.x = 0.08
                twist_cmd.angular.z = self.lidar_steer
                self.cmd_vel_pub.publish(twist_cmd)
                _, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="nearsighted")
            else:
                twist_cmd, final_view, _ = self.line_follower.process_image(cv_image, turn_direction=turn_dir)
                self._publish_with_delay(twist_cmd)

        # 🌟 Stage 5 停車關卡處理 (已完全對接動態視覺回饋)
        elif self.cs == 5:
            action, value = self.s5.get_action()
            
            if action == "cmd_vel":
                twist_cmd = Twist()
                twist_cmd.linear.x = float(value[0])
                twist_cmd.angular.z = float(value[1])
                self.cmd_vel_pub.publish(twist_cmd)
                # 依然維持擷取影像狀態，並把是否對齊的狀態傳給 Stage 5
                _, final_view, is_aligned = self.line_follower.process_image(cv_image, turn_direction="double_yellow")
                self.s5.process_vision(is_aligned)

            elif action == "spin":
                spin_cmd = Twist()
                spin_cmd.linear.x = 0.0     
                spin_cmd.angular.z = float(value)
                self.cmd_vel_pub.publish(spin_cmd)
                # 🌟 自轉時，眼睛要死盯著看「雙黃線」有沒有出現在畫面正中央！
                _, final_view, is_aligned = self.line_follower.process_image(cv_image, turn_direction="double_yellow")
                self.s5.process_vision(is_aligned) 
                
            elif action == "line_follow":
                twist_cmd, final_view, is_aligned = self.line_follower.process_image(cv_image, turn_direction=value)
                self._publish_with_delay(twist_cmd)
                self.s5.process_vision(is_aligned) # 🌟 視覺巡航時，持續回報眼睛狀態！
                
        # 🌟 Stage 6 柵欄關卡處理 
        elif self.cs == 6:
            # 第一步：把原始畫面餵給第六關，讓它去算畫面裡有多少「紅色像素」
            self.s6.process_vision(cv_image)
            
            # 第二步：詢問第六關現在該做什麼
            action, value = self.s6.get_action()
            
            if action == "cmd_vel":
                # 執行煞車指令 (停在柵欄前等 2 秒)
                twist_cmd = Twist()
                twist_cmd.linear.x = float(value[0])
                twist_cmd.angular.z = float(value[1])
                self.cmd_vel_pub.publish(twist_cmd)
                
                # 停車時依然畫出尋線畫面，這樣你 Debug 時才看得到最終畫面 final_view
                _, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="straight")
                
            elif action == "line_follow":
                # 正常循線 (靠近柵欄中，或是柵欄已經完全打開了)
                twist_cmd, final_view, _ = self.line_follower.process_image(cv_image, turn_direction=value)
                self._publish_with_delay(twist_cmd)

        # 🌟 Stage 7 黑盒子與終點線
        elif self.cs == 7:
            self.s7.process_vision(cv_image) # 讓它找紅線
            action, value = self.s7.get_action()
            
            if action == "cmd_vel":
                twist_cmd = Twist()
                twist_cmd.linear.x = float(value[0])
                twist_cmd.angular.z = float(value[1])
                self.cmd_vel_pub.publish(twist_cmd)
                _, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="straight")
                
            elif action == "line_follow":
                twist_cmd, final_view, _ = self.line_follower.process_image(cv_image, turn_direction=value)
                self._publish_with_delay(twist_cmd)

            elif action == "stop":
                self.get_logger().info("🏆 RACE FINISHED! STOPPING THE CAR!", throttle_duration_sec=1.0)
                twist_cmd = Twist()
                self.cmd_vel_pub.publish(twist_cmd)
                self.is_running = False # 🌟 切斷動力，完美收官！
                _, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="straight")

        if final_view is not None:
            try:
                img_msg = Image()
                img_msg.header.stamp = self.get_clock().now().to_msg()
                img_msg.height = final_view.shape[0]
                img_msg.width = final_view.shape[1]
                img_msg.encoding = 'bgr8'
                img_msg.is_bigendian = 0
                img_msg.step = final_view.shape[1] * 3
                img_msg.data = final_view.tobytes()
                self.line_view_pub.publish(img_msg)
            except Exception:
                pass

def main(args=None):
    rclpy.init(args=args)
    node = MainController()
    rclpy.spin(node)
    
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()