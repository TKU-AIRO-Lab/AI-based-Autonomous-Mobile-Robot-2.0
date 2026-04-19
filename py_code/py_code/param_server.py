"""
TurtleBot3 比賽用 - 即時參數調整伺服器 (v2)
============================================
速度參數已移除 (固定不調)，其餘所有可調參數保留。

啟動：
  pip3 install flask flask-cors
  python3 param_server.py
"""

import os
import sys
os.environ['ROS_DOMAIN_ID'] = '10'

# SLAM 地圖可視化（與 lidar_node 獨立的 TCP 連線）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from slam_visualizer import (
        start_viz,
        get_latest_frame  as get_slam_frame,
        get_nav_angle,
        set_custom_exit,
        clear_custom_exit,
        get_custom_exit,
        set_custom_start,
        clear_custom_start,
        get_custom_start,
        save_map_now,
        load_map,
        unload_map,
        get_map_list,
        delete_map,
        is_offline_mode,
    )
    _slam_viz_available = True
except ImportError as _e:
    print(f"[ParamServer] slam_visualizer 載入失敗: {_e}")
    _slam_viz_available = False
    def get_slam_frame():     return None
    def get_nav_angle():      return None
    def set_custom_exit(*a):  return None
    def clear_custom_exit():  pass
    def get_custom_exit():    return None
    def set_custom_start(*a): return None
    def clear_custom_start(): pass
    def get_custom_start():   return None
    def save_map_now(n):      return None
    def load_map(n):          return False
    def unload_map():         pass
    def get_map_list():       return []
    def delete_map(n):        return False
    def is_offline_mode():    return False

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Int32
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
        "conf_threshold": 0.65,
        "brightness": 0,
        "saturation_scale": 1.0,
    },
    "line": {
        "Kp": 0.002,
        "Kd": 0.00015,
        "base_speed": 0.1,
        "wheel_delay": 0.5,
        "line_mask_size": 0.0,
        "line_mask_top_w": 0.5,
        "undistort_enabled": 0,
        "yellow_h_low": 20, "yellow_s_low": 50, "yellow_v_low": 100,
        "yellow_h_high": 40, "yellow_s_high": 255, "yellow_v_high": 255,
        "white_h_low": 0, "white_s_low": 0, "white_v_low": 200,
        "white_h_high": 180, "white_s_high": 40, "white_v_high": 255,
        "brightness": 0,
        "saturation_scale": 1.0,
    },
    "s1": {},
    "s2": {
        "approach_duration": 7.5,
        "spin_speed": 0.5,
        "blind_spin_duration_left": 1.2,
        "blind_spin_duration_right": 0.70,
    },
    "s3": {
        "approach_duration": 2.0,
        "spin_speed": 0.5,
        "blind_spin_duration_left": 1.2,
        "blind_spin_duration_right": 0.70,
    },
    "s4": {
        "roi_top_ratio": 0.7,
        "danger_zone": 0.30,
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
        "red_threshold": 4000,
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
    "emergency_stop": 0,
}

# ============================================================
#  事件 log buffer（供 UI 顯示）
# ============================================================
_event_log = []
_event_log_lock = threading.Lock()

def push_log(msg, level="info"):
    """從任意執行緒安全地推送一筆 log 到 UI buffer。"""
    import time as _time
    ts = _time.strftime('%H:%M:%S')
    with _event_log_lock:
        _event_log.append({"t": ts, "msg": msg, "level": level})
        if len(_event_log) > 100:
            _event_log.pop(0)

# ============================================================
#  影像串流 buffer
# ============================================================
_stage_pub = None   # 由 ParamServerNode 初始化後設定，供 Flask 執行緒使用

latest_yolo_frame = None
latest_line_frame = None
latest_raw_frame  = None   # /image_raw 原始鏡頭（供黃/白線遮罩串流）
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

@app.route("/api/logs")
def get_logs():
    with _event_log_lock:
        return jsonify(list(_event_log))

SAVE_FILE = '/home/tkuai/tuning_params.json'

@app.route("/api/save_params", methods=["POST"])
def save_params():
    import time as _time
    try:
        record = {
            "saved_at": _time.strftime('%Y-%m-%d %H:%M:%S'),
            "params": shared_params
        }
        with open(SAVE_FILE, 'w', encoding='utf-8') as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        push_log(f"💾 參數已儲存至 {SAVE_FILE}", "stage")
        return jsonify({"status": "ok", "file": SAVE_FILE})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/emergency_stop", methods=["POST"])
def api_emergency_stop():
    shared_params["emergency_stop"] = 1
    push_log("⛔ 緊急停止", "error")
    return jsonify({"status": "ok", "emergency_stop": 1})

@app.route("/api/emergency_resume", methods=["POST"])
def api_emergency_resume():
    shared_params["emergency_stop"] = 0
    push_log("▶ 恢復行駛", "stage")
    return jsonify({"status": "ok", "emergency_stop": 0})

@app.route("/api/emergency_status", methods=["GET"])
def api_emergency_status():
    return jsonify({"emergency_stop": shared_params.get("emergency_stop", 0)})

@app.route("/api/set_stage", methods=["POST"])
def set_stage():
    data = request.get_json()
    if not data or 'stage' not in data:
        return jsonify({"error": "需要 stage 欄位"}), 400
    if _stage_pub is None:
        return jsonify({"error": "ROS node 尚未就緒"}), 503
    msg = Int32()
    msg.data = int(data['stage'])
    _stage_pub.publish(msg)
    return jsonify({"status": "ok", "stage": msg.data})

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

def gen_color_mask(color: str):
    """黃線/白線 HSV 遮罩即時串流（從 /image_raw 抓原始畫面，套用 shared_params HSV）"""
    blank_jpg = None
    while True:
        raw = None
        with frame_lock:
            raw = latest_raw_frame
        if raw is not None:
            try:
                p = shared_params.get("line", {})
                # 亮度/飽和度預處理（與 line.py 一致）
                brightness  = int(p.get("brightness", 0))
                sat_scale   = float(p.get("saturation_scale", 1.0))
                img = raw.copy()
                if brightness != 0 or sat_scale != 1.0:
                    h_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
                    h_img[:, :, 1] = np.clip(h_img[:, :, 1] * sat_scale, 0, 255)
                    h_img[:, :, 2] = np.clip(h_img[:, :, 2] + brightness, 0, 255)
                    img = cv2.cvtColor(h_img.astype(np.uint8), cv2.COLOR_HSV2BGR)
                if color == "yellow":
                    lo = np.array([p.get("yellow_h_low",  20),  p.get("yellow_s_low",  100), p.get("yellow_v_low",  100)], dtype=np.uint8)
                    hi = np.array([p.get("yellow_h_high", 40),  p.get("yellow_s_high", 255), p.get("yellow_v_high", 255)], dtype=np.uint8)
                    label, clr = "YELLOW", (0, 220, 220)
                else:
                    lo = np.array([p.get("white_h_low",   0),   p.get("white_s_low",   0),   p.get("white_v_low",   200)], dtype=np.uint8)
                    hi = np.array([p.get("white_h_high",  180), p.get("white_s_high",  50),  p.get("white_v_high",  255)], dtype=np.uint8)
                    label, clr = "WHITE",  (200, 200, 200)
                hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lo, hi)
                result = cv2.bitwise_and(img, img, mask=mask)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, clr, 1)
                cv2.putText(result, label, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 1)
                _, jpg = cv2.imencode('.jpg', result, [cv2.IMWRITE_JPEG_QUALITY, 70])
                yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg.tobytes() + b'\r\n'
            except Exception:
                pass
        else:
            if blank_jpg is None:
                blank = np.zeros((240, 320, 3), dtype=np.uint8)
                lbl = "yellow" if color == "yellow" else "white"
                cv2.putText(blank, f"Waiting for {lbl} mask...", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
                _, buf = cv2.imencode('.jpg', blank)
                blank_jpg = buf.tobytes()
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank_jpg + b'\r\n'
        threading.Event().wait(0.2)

@app.route("/stream/yellow")
def stream_yellow():
    return Response(gen_color_mask("yellow"), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stream/white")
def stream_white():
    return Response(gen_color_mask("white"), mimetype="multipart/x-mixed-replace; boundary=frame")

def gen_slam_mjpeg():
    """SLAM 地圖 MJPEG 串流（每秒 1 幀，由 slam_visualizer 背景執行緒更新）"""
    blank_jpg = None
    while True:
        frame = get_slam_frame()
        if frame is not None:
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            if blank_jpg is None:
                import numpy as _np
                blank = _np.zeros((240, 320, 3), dtype=_np.uint8)
                cv2.putText(blank, "Waiting for SLAM map...", (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
                _, buf = cv2.imencode('.jpg', blank)
                blank_jpg = buf.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + blank_jpg + b'\r\n')
        threading.Event().wait(1.0)

@app.route("/stream/map")
def stream_map():
    return Response(gen_slam_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/popout/map")
def popout_map():
    return Response('''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>SLAM 地圖</title>
<style>*{margin:0;padding:0;box-sizing:border-box}body{background:#0b0f19;display:flex;flex-direction:column;align-items:center;height:100vh}
.bar{width:100%;padding:6px 10px;background:#121a2b;color:#2d9cdb;font-family:monospace;font-size:12px;font-weight:600;letter-spacing:1px;display:flex;justify-content:space-between}
.bar span{font-size:10px;color:#7b8ba5}
img{flex:1;max-width:100%;max-height:calc(100vh - 30px);object-fit:contain}</style></head>
<body>
<div class="bar">SLAM 地圖 <span>橘圓=機器人 | 紅線=A*最佳路徑 | 紅星=出口</span></div>
<img src="/stream/map" alt="SLAM Map">
</body></html>''', mimetype='text/html')

# ============================================================
#  SLAM 獨立頁面 + API
# ============================================================
@app.route("/slam")
def slam_page():
    slam_html = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'slam_panel.html')
    slam_html = os.path.normpath(slam_html)
    if os.path.exists(slam_html):
        return send_file(slam_html)
    return Response("<h2>找不到 slam_panel.html</h2>", mimetype='text/html'), 404

@app.route("/stream/slam")
def stream_slam():
    """SLAM 地圖 MJPEG 串流（/stream/map 的別名，供 slam_panel.html 使用）"""
    return Response(gen_slam_mjpeg(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/slam/status")
def slam_status():
    cs_val = get_custom_start()
    ce_val = get_custom_exit()
    return jsonify({
        "nav_angle":    get_nav_angle(),
        "offline_mode": is_offline_mode(),
        "custom_start": list(cs_val) if cs_val else None,
        "custom_exit":  list(ce_val) if ce_val else None,
        "maps":         get_map_list(),
    })

@app.route("/api/slam/set_exit", methods=["POST"])
def slam_set_exit():
    data = request.get_json() or {}
    px = int(data.get("x", 0))
    py = int(data.get("y", 0))
    result = set_custom_exit(px, py)
    if result is None:
        return jsonify({"error": "座標超出地圖範圍或地圖尚未載入"}), 400
    push_log(f"SLAM 自訂出口設定：格子 {result}", "stage")
    return jsonify({"status": "ok", "grid": list(result)})

@app.route("/api/slam/clear_exit", methods=["POST"])
def slam_clear_exit():
    clear_custom_exit()
    push_log("SLAM 出口已清除，回到自動偵測模式", "info")
    return jsonify({"status": "ok"})

@app.route("/api/slam/set_start", methods=["POST"])
def slam_set_start():
    data = request.get_json() or {}
    px = int(data.get("x", 0))
    py = int(data.get("y", 0))
    result = set_custom_start(px, py)
    if result is None:
        return jsonify({"error": "座標超出地圖範圍或地圖尚未載入"}), 400
    push_log(f"SLAM 自訂起點設定：格子 {result}", "stage")
    return jsonify({"status": "ok", "grid": list(result)})

@app.route("/api/slam/clear_start", methods=["POST"])
def slam_clear_start():
    clear_custom_start()
    push_log("SLAM 起點已清除，回到機器人位姿模式", "info")
    return jsonify({"status": "ok"})

@app.route("/api/slam/save_map", methods=["POST"])
def slam_save_map():
    data = request.get_json() or {}
    name = str(data.get("name", "map")).strip() or "map"
    path = save_map_now(name)
    if path is None:
        return jsonify({"error": "儲存失敗，請確認 M2M2 已連線或已載入地圖"}), 500
    push_log(f"SLAM 地圖已儲存：{name}", "stage")
    return jsonify({"status": "ok", "path": path})

@app.route("/api/slam/load_map", methods=["POST"])
def slam_load_map():
    data = request.get_json() or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"error": "請提供地圖名稱"}), 400
    ok = load_map(name)
    if not ok:
        return jsonify({"error": f"找不到地圖：{name}"}), 404
    push_log(f"SLAM 地圖已載入（離線模式）：{name}", "stage")
    return jsonify({"status": "ok", "offline_mode": True})

@app.route("/api/slam/unload_map", methods=["POST"])
def slam_unload_map():
    unload_map()
    push_log("SLAM 切回即時模式", "info")
    return jsonify({"status": "ok", "offline_mode": False})

@app.route("/api/slam/maps")
def slam_maps():
    return jsonify(get_map_list())

@app.route("/api/slam/delete_map", methods=["POST"])
def slam_delete_map():
    data = request.get_json() or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"error": "請提供地圖名稱"}), 400
    ok = delete_map(name)
    if not ok:
        return jsonify({"error": f"找不到地圖：{name}"}), 404
    push_log(f"SLAM 地圖已刪除：{name}", "info")
    return jsonify({"status": "ok"})

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
    const px=cx+rng*sc*Math.sin(a),py=cy+rng*sc*Math.cos(a);
    ctx.fillRect(px-1.5,py-1.5,3,3);
  }
  ctx.fillStyle='#2d9cdb';ctx.beginPath();ctx.arc(cx,cy,5,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#2d9cdb';ctx.beginPath();ctx.moveTo(cx,cy+14);ctx.lineTo(cx-5,cy+6);ctx.lineTo(cx+5,cy+6);ctx.fill();
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
def _free_port(port: int):
    """Kill any process occupying the given port before binding."""
    import signal, subprocess
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f":{port}"], stderr=subprocess.DEVNULL
        ).decode().split()
        own_pid = str(os.getpid())
        for pid in out:
            if pid != own_pid:
                os.kill(int(pid), signal.SIGTERM)
        if out:
            import time; time.sleep(2.0)   # 等 port + DDS 資源釋放
    except subprocess.CalledProcessError:
        pass  # 沒有進程佔用，正常

class ParamServerNode(Node):
    def __init__(self):
        global _stage_pub
        super().__init__("param_server")
        self.param_pub = self.create_publisher(String, "/tuning_params", 10)
        self.stage_pub = self.create_publisher(Int32, "/set_stage", 10)
        _stage_pub = self.stage_pub
        self.timer = self.create_timer(2.0, self.publish_params)
        self.yolo_img_sub = self.create_subscription(Image, '/yolo_image', self.yolo_image_cb, 10)
        self.line_img_sub = self.create_subscription(Image, '/line_view',  self.line_image_cb, 10)
        self.raw_img_sub  = self.create_subscription(Image, '/image_raw',  self.raw_image_cb,  10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_cb, 10)
        self.flask_thread = threading.Thread(
            target=lambda: app.run(host="0.0.0.0", port=5000, debug=False), daemon=True)
        self.flask_thread.start()
        # 啟動 SLAM 地圖可視化背景執行緒
        if _slam_viz_available:
            start_viz()
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

    def raw_image_cb(self, msg):
        global latest_raw_frame
        try:
            img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            if msg.encoding == 'rgb8': img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            with frame_lock: latest_raw_frame = img
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
    _free_port(5000)   # 先清舊進程，再初始化 ROS context
    rclpy.init(args=args)
    node = ParamServerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
