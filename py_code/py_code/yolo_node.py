import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
import numpy as np
import cv2
import json
import os
import torch
from ultralytics import YOLO
from .param_server import shared_params

# ── 相機校正（與 main_controller 共用同一份 calibration.npz）────────────────
_CALIB_PATH = os.path.expanduser('~/ros2_ws/calibration.npz')

def _load_calibration():
    if not os.path.exists(_CALIB_PATH):
        return None, None
    data = np.load(_CALIB_PATH)
    return data['mtx'], data['dist']

_CAM_MTX, _CAM_DIST = _load_calibration()
_CAM_MAPS = {}

def undistort(frame):
    if _CAM_MTX is None:
        return frame
    h, w = frame.shape[:2]
    if (h, w) not in _CAM_MAPS:
        new_mtx, _ = cv2.getOptimalNewCameraMatrix(_CAM_MTX, _CAM_DIST, (w, h), 0, (w, h))
        map1, map2 = cv2.initUndistortRectifyMap(
            _CAM_MTX, _CAM_DIST, None, new_mtx, (w, h), cv2.CV_16SC2
        )
        _CAM_MAPS[(h, w)] = (map1, map2)
    map1, map2 = _CAM_MAPS[(h, w)]
    return cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
# ─────────────────────────────────────────────────────────────────────────────

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_node')
        
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        self.params_sub = self.create_subscription(
            String,
            '/tuning_params',
            self.params_callback,
            10)

        self.publisher_ = self.create_publisher(String, '/yolo', 10)
        self.image_pub_ = self.create_publisher(Image, '/yolo_image', 10)

        # 檢查 CUDA
        self.get_logger().info(f"CUDA available: {torch.cuda.is_available()}")
        self.get_logger().info(f"CUDA device count: {torch.cuda.device_count()}")
        
        
        # YOLO 模型路徑
        model_path = '/home/tkuai/ros2_ws/src/py_code/weights/best.pt'
        self.get_logger().info(f"Loading YOLO model from {model_path} ...")
        self.model = YOLO(model_path) 
        self.model.to('cuda')  # 強制用 GPU
        self.get_logger().info(f"Model running on: {next(self.model.model.parameters()).device}")
        self.get_logger().info("YOLO Model loaded successfully.")

    def params_callback(self, msg):
        try:
            new_params = json.loads(msg.data)
            shared_params.update(new_params)
        except Exception:
            pass

    def image_callback(self, msg):
        p = shared_params.get("yolo", {})
        brightness = p.get("brightness", 0)
        sat_scale = p.get("saturation_scale", 1.0)
        conf = p.get("conf_threshold", 0.25)
 
        try:
            # 把 ROS Image 轉成 NumPy Array
            img_array = np.ndarray(shape=(msg.height, msg.width, 3), dtype=np.uint8, buffer=msg.data)
            
            if msg.encoding == 'rgb8':
                cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            elif msg.encoding == 'bgr8':
                cv_image = img_array
            else:
                cv_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            cv_image = undistort(cv_image)

        except Exception as e:
            self.get_logger().error(f"Failed to convert image manually: {e}")
            return

        if brightness != 0 or sat_scale != 1.0:
            hsv_adj = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_adj[:,:,1] *= sat_scale
            hsv_adj[:,:,2] += brightness
            hsv_adj = np.clip(hsv_adj, 0, 255).astype(np.uint8)
            cv_image = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2BGR)


        # 執行YOLO辨識
        results = self.model(cv_image, verbose=False, conf=conf)
        
        # 取得畫好框框的圖片
        annotate_frame = results[0].plot()

        # 解析偵測到的標籤
        detect_sign = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                detect_sign.append(class_name)

        if detect_sign:
            msg_out = String()
            msg_out.data = ",".join(detect_sign) 
            self.publisher_.publish(msg_out)

        try:
            img_msg = Image()
            img_msg.header = msg.header
            img_msg.height = annotate_frame.shape[0]
            img_msg.width = annotate_frame.shape[1]
            img_msg.encoding = 'bgr8'
            img_msg.is_bigendian = 0
            img_msg.step = annotate_frame.shape[1] * 3
            img_msg.data = annotate_frame.tobytes()
            self.image_pub_.publish(img_msg)
        except Exception as e:
            self.get_logger().error(f"Failed to publish yolo image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()