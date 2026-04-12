"""
TurtleBot3 比賽用 - 即時參數調整伺服器 (v2)
============================================
速度參數已移除 (固定不調)，其餘所有可調參數保留。

啟動：
  pip3 install flask flask-cors
  python3 param_server.py
"""

import os
os.environ.setdefault('ROS_DOMAIN_ID', '10')

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image, LaserScan
import json
import math
import threading
import cv2
import numpy as np
from flask import Flask, jsonify, request, Response, send_file
from flask_cors import CORS

# ============================================================
#  全域共用參數
# ============================================================
shared_params = {
    "yolo": {
        "conf_threshold": 0.25,
        "brightness": 0,
        "saturation_scale": 1.0,
    },
    "line": {
        "Kp": 0.003,
        "line_mask_size": 0.5,
        "line_mask_top_w": 0.5,
        "yellow_h_low": 20, "yellow_s_low": 100, "yellow_v_low": 100,
        "yellow_h_high": 40, "yellow_s_high": 255, "yellow_v_high": 255,
        "white_h_low": 0, "white_s_low": 0, "white_v_low": 200,
        "white_h_high": 180, "white_s_high": 50, "white_v_high": 255,
        "brightness": 0,
        "saturation_scale": 1.0,
    },
    "s1": {},
    "s2": {
        "approach_duration": 4.5,
        "spin_speed": 0.5,
        "blind_spin_duration_left": 1.57,
        "blind_spin_duration_right": 0.70,
    },
    "s3": {
        "approach_duration": 2.0,
        "spin_speed": 0.5,
        "blind_spin_duration_left": 1.57,
        "blind_spin_duration_right": 0.70,
    },
    "s4": {
        "roi_top_ratio": 0.7,
        "danger_zone": 0.35,
        "avoid_steer": 0.8,
        "avoid_duration": 5.0,
    },
    "s5": {
        "forward_delay": 5.0,
        "turn_in_duration": 1.57,
        "drive_in_duration": 2.0,
        "parked_duration": 2.0,
        "drive_out_duration": 2.0,
        "spin_out_duration": 1.57,
        "empty_threshold": 0.45,
        "lost_line_timeout": 0.5,
        "blind_forward_duration": 2.0,
        "final_turn_duration": 1.57,
    },
    "s6": {
        "red_threshold": 2000,
        "debounce_duration": 2.0,
        "red_h_low_1": 0, "red_s_low_1": 70, "red_v_low_1": 50,
        "red_h_high_1": 10, "red_s_high_1": 255, "red_v_high_1": 255,
        "red_h_low_2": 170, "red_s_low_2": 70, "red_v_low_2": 50,
        "red_h_high_2": 180, "red_s_high_2": 255, "red_v_high_2": 255,
    },
    "s7": {
        "blind_forward_duration": 2.5,
        "nav_Kp": 0.012,
        "emergency_distance": 0.35,
        "emergency_steer": 0.8,
        "red_threshold": 200,
        "exit_duration": 3.0,
        "lost_line_timeout": 0.5,
    },
}

# ============================================================
#  影像串流 buffer
# ============================================================
latest_yolo_frame = None
latest_line_frame = None
frame_lock = threading.Lock()

# ============================================================
#  光達資料 buffer
# ============================================================
latest_scan_data = None
scan_lock = threading.Lock()

# ============================================================
#  Flask
# ============================================================
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return send_file('/home/tkuai/Downloads/tuning_panel.html')

@app.route("/api/params", methods=["GET"])
def get_all_params():
    return jsonify(shared_params)

@app.route("/api/params/<stage>", methods=["GET"])
def get_stage_params(stage):
    if stage in shared_params:
        return jsonify(shared_params[stage])
    return jsonify({"error": f"Unknown stage: {stage}"}), 404

@app.route("/api/params/<stage>", methods=["POST"])
def update_stage_params(stage):
    if stage not in shared_params:
        return jsonify({"error": f"Unknown stage: {stage}"}), 404
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400
    for key, value in data.items():
        if key in shared_params[stage]:
            orig = shared_params[stage][key]
            if isinstance(orig, float):
                shared_params[stage][key] = float(value)
            elif isinstance(orig, int):
                shared_params[stage][key] = int(value)
            else:
                shared_params[stage][key] = value
    return jsonify({"status": "ok", "updated": shared_params[stage]})

def gen_mjpeg(source):
    while True:
        frame = None
        with frame_lock:
            frame = latest_yolo_frame if source == "yolo" else latest_line_frame
        if frame is not None:
            _, jpg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        else:
            blank = np.zeros((1, 1, 3), dtype=np.uint8)
            _, jpg = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n')
        threading.Event().wait(0.05)

@app.route("/stream/yolo")
def stream_yolo():
    return Response(gen_mjpeg("yolo"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream/line")
def stream_line():
    return Response(gen_mjpeg("line"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/scan")
def get_scan():
    with scan_lock:
        if latest_scan_data is None:
            return jsonify({"error": "no data"}), 503
        return jsonify(latest_scan_data)

@app.route("/popout/yolo")
def popout_yolo():
    return Response('''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>YOLO 辨識</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#000;display:flex;flex-direction:column;align-items:center;height:100vh}
.bar{width:100%;padding:6px 10px;background:#121a2b;color:#2d9cdb;font-family:monospace;font-size:12px;font-weight:600;letter-spacing:1px}
img{flex:1;max-width:100%;max-height:calc(100vh - 30px);object-fit:contain}</style></head>
<body><div class="bar">YOLO 辨識畫面</div><img src="/stream/yolo" alt="YOLO"></body></html>''', mimetype='text/html')

@app.route("/popout/line")
def popout_line():
    return Response('''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>循線 Debug</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#000;display:flex;flex-direction:column;align-items:center;height:100vh}
.bar{width:100%;padding:6px 10px;background:#121a2b;color:#2d9cdb;font-family:monospace;font-size:12px;font-weight:600;letter-spacing:1px}
img{flex:1;max-width:100%;max-height:calc(100vh - 30px);object-fit:contain}</style></head>
<body><div class="bar">循線 Debug 畫面</div><img src="/stream/line" alt="Line"></body></html>''', mimetype='text/html')

@app.route("/popout/lidar")
def popout_lidar():
    return Response('''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>光達掃描</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0b0f19;display:flex;flex-direction:column;align-items:center;height:100vh;overflow:hidden}
.bar{width:100%;padding:6px 10px;background:#121a2b;color:#2d9cdb;font-family:monospace;font-size:12px;font-weight:600;letter-spacing:1px;display:flex;justify-content:space-between;align-items:center}
.bar .st{font-size:10px;color:#7b8ba5}
canvas{flex:1;max-width:100%;max-height:calc(100vh - 30px)}
</style></head>
<body>
<div class="bar">光達掃描 <span class="st" id="info">等待資料...</span></div>
<canvas id="c"></canvas>
<script>
const canvas=document.getElementById('c');
const ctx=canvas.getContext('2d');
let scan=null;
function resize(){canvas.width=canvas.offsetWidth;canvas.height=canvas.offsetHeight}
window.addEventListener('resize',()=>{resize();draw()});
resize();
function draw(){
  if(!scan){ctx.fillStyle='#0b0f19';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='#7b8ba5';ctx.font='14px monospace';ctx.textAlign='center';
    ctx.fillText('等待光達資料...',canvas.width/2,canvas.height/2);return}
  const W=canvas.width,H=canvas.height,cx=W/2,cy=H/2;
  const R=Math.min(cx,cy)-30;
  const rMax=Math.min(scan.range_max||5,8);
  const sc=R/rMax;
  ctx.fillStyle='#0b0f19';ctx.fillRect(0,0,W,H);
  const step=rMax<=3?0.5:1;
  for(let r=step;r<=rMax;r+=step){
    ctx.strokeStyle='#263048';ctx.lineWidth=0.5;
    ctx.beginPath();ctx.arc(cx,cy,r*sc,0,Math.PI*2);ctx.stroke();
    ctx.fillStyle='#4a5568';ctx.font='9px monospace';ctx.textAlign='left';
    ctx.fillText(r.toFixed(1)+'m',cx+r*sc+3,cy-2);
  }
  ctx.strokeStyle='#1e2a3a';ctx.lineWidth=0.5;
  ctx.beginPath();ctx.moveTo(cx,cy-R);ctx.lineTo(cx,cy+R);ctx.stroke();
  ctx.beginPath();ctx.moveTo(cx-R,cy);ctx.lineTo(cx+R,cy);ctx.stroke();
  const dz=scan.danger_zone||0.35;
  ctx.strokeStyle='rgba(231,76,60,0.7)';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.arc(cx,cy,dz*sc,0,Math.PI*2);ctx.stroke();
  ctx.strokeStyle='rgba(45,156,219,0.4)';ctx.lineWidth=1;ctx.setLineDash([4,4]);
  ctx.beginPath();ctx.moveTo(cx,cy-R);ctx.lineTo(cx,cy);ctx.stroke();
  ctx.setLineDash([]);
  const{ranges,angle_min,angle_increment,range_min,range_max}=scan;
  ctx.fillStyle='#27ae60';
  for(let i=0;i<ranges.length;i++){
    const rng=ranges[i];
    if(rng<=0||!isFinite(rng)||rng<range_min||rng>range_max)continue;
    const a=angle_min+i*angle_increment;
    const px=cx+rng*sc*Math.sin(a),py=cy-rng*sc*Math.cos(a);
    ctx.fillRect(px-1.5,py-1.5,3,3);
  }
  ctx.fillStyle='#2d9cdb';ctx.beginPath();ctx.arc(cx,cy,5,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#2d9cdb';ctx.beginPath();ctx.moveTo(cx,cy-14);ctx.lineTo(cx-5,cy-6);ctx.lineTo(cx+5,cy-6);ctx.fill();
}
async function poll(){
  try{const r=await fetch('/api/scan');
    if(r.ok){scan=await r.json();
      const n=scan.ranges?scan.ranges.filter(x=>x>0&&isFinite(x)).length:0;
      document.getElementById('info').textContent=n+' 點 | max '+((scan.range_max||0).toFixed(1))+'m';
      draw();
    }
  }catch(e){}
  setTimeout(poll,100);
}
poll();
</script></body></html>''', mimetype='text/html')

# ============================================================
#  ROS2 Node
# ============================================================
class ParamServerNode(Node):
    def __init__(self):
        super().__init__("param_server")
        self.param_pub = self.create_publisher(String, "/tuning_params", 10)
        self.timer = self.create_timer(0.5, self.publish_params)
        self.yolo_img_sub = self.create_subscription(Image, '/yolo_image', self.yolo_image_cb, 10)
        self.line_img_sub = self.create_subscription(Image, '/line_view', self.line_image_cb, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.flask_thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=5000, debug=False), daemon=True)
        self.flask_thread.start()
        self.get_logger().info("Param Server ready at http://0.0.0.0:5000")

    def publish_params(self):
        msg = String()
        msg.data = json.dumps(shared_params)
        self.param_pub.publish(msg)

    def yolo_image_cb(self, msg):
        global latest_yolo_frame
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8': img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            with frame_lock: latest_yolo_frame = img
        except: pass

    def line_image_cb(self, msg):
        global latest_line_frame
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8': img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            with frame_lock: latest_line_frame = img
        except: pass

    def scan_cb(self, msg):
        global latest_scan_data
        ranges = [r if math.isfinite(r) and r > 0 else 0.0 for r in msg.ranges]
        with scan_lock:
            latest_scan_data = {
                "ranges": ranges,
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                "range_min": float(msg.range_min),
                "range_max": float(msg.range_max),
            }

def main(args=None):
    rclpy.init(args=args)
    node = ParamServerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
