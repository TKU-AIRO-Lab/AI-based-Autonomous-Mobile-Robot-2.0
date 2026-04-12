"""
Camera calibration node for ROS2.

Usage:
  ros2 run py_code camera_calibration_node

  # 瀏覽器開啟看即時畫面（同一台機器或同網段）：
  http://<robot-ip>:5001

棋盤格偵測成功 → 畫面顯示綠色角點，自動存圖（每 1.5 秒最多一張）
收集到 TARGET_FRAMES 張後自動計算並存到 SAVE_PATH

Chessboard: 9x7 inner corners (10x8 squares)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
import time
import os
import threading
from flask import Flask, Response

# ── 設定區 ────────────────────────────────────────────────────────────────────
CHECKERBOARD = (6, 8)          # 棋盤格內角點數量 (columns, rows)，格子數 7x9 → 內角點 6x8
TARGET_FRAMES = 30             # 收集幾張後開始計算
MIN_INTERVAL_SEC = 1.5         # 兩次成功存圖的最短間隔（秒）
SAVE_PATH = os.path.expanduser('~/ros2_ws/calibration.npz')
# ─────────────────────────────────────────────────────────────────────────────

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ── Flask MJPEG 串流 ──────────────────────────────────────────────────────────
_latest_frame_jpg: bytes = b''
_frame_lock = threading.Lock()

_app = Flask(__name__)

@_app.route('/')
def _index():
    return (
        '<html><body style="background:#111;margin:0">'
        '<img src="/stream" style="width:100%;max-width:900px;display:block;margin:auto">'
        '</body></html>'
    )

@_app.route('/stream')
def _stream():
    def gen():
        while True:
            with _frame_lock:
                jpg = _latest_frame_jpg
            if jpg:
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n'
            time.sleep(0.05)
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

def _start_flask(port=5001):
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    _app.run(host='0.0.0.0', port=port, threaded=True)
# ─────────────────────────────────────────────────────────────────────────────


class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')

        self.objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

        self.objpoints = []
        self.imgpoints = []
        self.img_shape = None
        self.last_capture_time = 0.0
        self.done = False

        self.sub = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        self.debug_pub = self.create_publisher(Image, '/calibration_view', 10)

        # 背景啟動 Flask
        t = threading.Thread(target=_start_flask, daemon=True)
        t.start()

        self.get_logger().info(
            f'Calibration node ready. Collecting {TARGET_FRAMES} frames...\n'
            f'  --> Open browser: http://<robot-ip>:5001'
        )

    # ── 將 ROS Image msg 轉成 BGR numpy array ──────────────────────────────────
    def _msg_to_bgr(self, msg):
        if msg.encoding == 'rgb8':
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        elif msg.encoding == 'bgr8':
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            return arr.copy()
        elif msg.encoding in ('mono8', '8UC1'):
            arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        else:
            raise ValueError(f'Unsupported encoding: {msg.encoding}')

    # ── 把 BGR numpy array 打包成 ROS Image msg ────────────────────────────────
    def _bgr_to_msg(self, frame):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = 'bgr8'
        msg.is_bigendian = 0
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        return msg

    # ── 在 debug 畫面上疊加文字 ────────────────────────────────────────────────
    def _draw_status(self, frame, detected, count):
        vis = frame.copy()
        h, w = vis.shape[:2]

        # 進度條背景
        cv2.rectangle(vis, (0, 0), (w, 50), (0, 0, 0), -1)

        if self.done:
            text = 'DONE! calibration.npz saved.'
            color = (0, 255, 0)
        elif detected:
            pct = int(count / TARGET_FRAMES * (w - 20))
            cv2.rectangle(vis, (10, 10), (10 + pct, 40), (0, 200, 0), -1)
            text = f'DETECTED  [{count}/{TARGET_FRAMES}]  Move slowly...'
            color = (0, 255, 0)
        else:
            text = f'NOT DETECTED [{count}/{TARGET_FRAMES}]  Show full chessboard'
            color = (0, 80, 255)

        cv2.putText(vis, text, (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(vis, text, (10, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        return vis

    # ── 主 callback ────────────────────────────────────────────────────────────
    def image_callback(self, msg):
        try:
            frame = self._msg_to_bgr(msg)
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
            return

        self.img_shape = (frame.shape[1], frame.shape[0])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        count = len(self.objpoints)

        flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
                 + cv2.CALIB_CB_NORMALIZE_IMAGE
                 + cv2.CALIB_CB_FAST_CHECK)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

        if ret:
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            # 畫出角點（綠色）
            cv2.drawChessboardCorners(frame, CHECKERBOARD, corners_refined, ret)

            now = time.time()
            if not self.done and now - self.last_capture_time >= MIN_INTERVAL_SEC:
                self.objpoints.append(self.objp.copy())
                self.imgpoints.append(corners_refined)
                self.last_capture_time = now
                count += 1
                self.get_logger().info(f'[{count}/{TARGET_FRAMES}] Frame captured!')
                if count >= TARGET_FRAMES:
                    self._calibrate()

        vis = self._draw_status(frame, ret, count)
        self.debug_pub.publish(self._bgr_to_msg(vis))

        # 推到 Flask MJPEG 串流
        global _latest_frame_jpg
        ok, jpg_buf = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with _frame_lock:
                _latest_frame_jpg = jpg_buf.tobytes()

    # ── 計算校正 ───────────────────────────────────────────────────────────────
    def _calibrate(self):
        self.done = True
        self.get_logger().info('Calculating calibration...')

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, self.img_shape, None, None
        )

        if not ret:
            self.get_logger().error('Calibration FAILED!')
            return

        total_error = 0.0
        for i in range(len(self.objpoints)):
            projected, _ = cv2.projectPoints(
                self.objpoints[i], rvecs[i], tvecs[i], mtx, dist
            )
            total_error += cv2.norm(self.imgpoints[i], projected, cv2.NORM_L2) / len(projected)
        mean_error = total_error / len(self.objpoints)

        np.savez(SAVE_PATH, mtx=mtx, dist=dist)

        self.get_logger().info('=' * 50)
        self.get_logger().info(f'SUCCESS! Saved to {SAVE_PATH}')
        self.get_logger().info(f'Reprojection error: {mean_error:.4f} px  (good if < 1.0)')
        self.get_logger().info(f'Camera matrix:\n{mtx}')
        self.get_logger().info(f'Distortion coefficients: {dist.ravel()}')
        self.get_logger().info('=' * 50)
        self.get_logger().info('Restart main_controller to apply undistortion.')


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
