import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String

# ==========================================
# 1. 給 Master Controller 使用的核心邏輯模組
# ==========================================
class TrafficLightStage:
    def __init__(self):
        # 預設狀態 (Default State): Wait/Stop
        self.is_green_light_detected = False

    def check_start_condition(self, yolo_detection):
        """
        輸入：YOLO 辨識結果字串
        輸出：布林值 (True 代表可以起步，False 代表繼續等)
        """
        # 防呆機制：確保字串轉小寫並去除空白
        detection = str(yolo_detection).strip().lower()

        # 根據你的規則：過濾黃燈，只看綠燈
        if 'green light' in detection:
            self.is_green_light_detected = True
            
        return self.is_green_light_detected

    def get_stop_twist(self):
        """回傳一個全為 0 的 Twist 指令，確保車子不動"""
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        return twist

# ==========================================
# 2. 獨立測試專用節點 (Standalone Test Node)
# ==========================================
class Stage1TestNode(Node):
    def __init__(self):
        super().__init__('stage_1_test_node')
        self.stage_logic = TrafficLightStage()
        
        # 訂閱 YOLO 的辨識結果
        self.yolo_sub = self.create_subscription(
            String, '/yolo', self.yolo_callback, 10)
            
        # 發布車輪控制
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.get_logger().info("🛑 Stage 1 Test: Waiting for GREEN light to proceed...")

    def yolo_callback(self, msg):
        detection = msg.data
        self.get_logger().info(f"YOLO saw: {detection}")
        
        # 丟給核心邏輯判斷
        can_go = self.stage_logic.check_start_condition(detection)
        
        if can_go:
            self.get_logger().info("🟢 GREEN LIGHT! (If this were the real race, we would move to Stage 2)")
            # 測試模式下，我們就發布一個微微前進的速度代表成功
            twist = Twist()
            twist.linear.x = 0.05 
            self.cmd_vel_pub.publish(twist)
        else:
            self.get_logger().info("🛑 Still waiting... holding position.")
            # 繼續保持靜止
            stop_twist = self.stage_logic.get_stop_twist()
            self.cmd_vel_pub.publish(stop_twist)



def main(args=None):
    rclpy.init(args=args)
    node = Stage1TestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()