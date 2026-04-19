"""
SLAM 地圖可視化模組
===================
- 透過 TCP 直接連 M2M2 Mapper，取得 SLAM 佔用格地圖與機器人位姿
- 使用 A* 計算從目前位置到出口的最短路徑
- 輸出 JPEG bytes 供 Flask MJPEG 串流

架構：與 lidar_node 的 scan 連線完全獨立（各自用一條 TCP socket）
"""

import socket
import json
import base64
import math
import threading
import time
import pathlib
import numpy as np
import cv2
import heapq

MAPS_DIR = pathlib.Path.home() / "maps"

LIDAR_HOST = '192.168.11.1'
LIDAR_PORT = 1445

# ── 執行緒共享資料 ────────────────────────────────────────────
_latest_frame: bytes | None = None
_frame_lock = threading.Lock()
_worker_thread: threading.Thread | None = None
_running = False

# ── 供 S7 狀態機顯示目前導航角度（由 slam_visualizer 更新）──
latest_nav_angle: float = 0.0

# ── A* 導航角度（供 s7_tunnel 實際使用）──
_slam_nav: dict = {"angle": None, "ts": 0.0}
_nav_lock  = threading.Lock()

# ── 自訂出口點（格子座標，y 已翻轉對應 disp 方向）──
_custom_exit: tuple[int, int] | None = None
_custom_exit_lock = threading.Lock()

# ── 自訂起點（格子座標，y 已翻轉對應 disp 方向）──
_custom_start: tuple[int, int] | None = None
_custom_start_lock = threading.Lock()

# ── 地圖尺寸快取（供像素→格子座標換算）──
_map_meta: dict = {"w": 0, "h": 0, "scale": 1}

# ── 離線地圖資料（載入後使用，優先於即時 getmap）──
_offline_map: dict | None = None   # {"map": {...}, "pose": {...}, "custom_exit": ...}
_offline_lock = threading.Lock()


# ─────────────────────────────────────────────────────────────
#  M2M2 通訊
# ─────────────────────────────────────────────────────────────
class SlamMapper:
    """精簡版 M2M2 TCP 客戶端（地圖 + 位姿）"""

    def __init__(self, host: str = LIDAR_HOST, port: int = LIDAR_PORT):
        self.host = host
        self.port = port
        self._sock: socket.socket | None = None
        self._req_id = 0

    def connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(10.0)
        self._sock.connect((self.host, self.port))

    def disconnect(self):
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send(self, command: str, args=None) -> dict:
        req = {"command": command, "args": args, "request_id": self._req_id}
        self._req_id += 1
        raw = json.dumps(req)
        payload = bytearray(ord(c) for c in raw)
        payload.extend([10, 13, 10, 13, 10])  # M2M2 結尾換行符號
        self._sock.sendall(payload)

        buf = b""
        while True:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Socket closed by remote")
            buf += chunk
            if buf[-4:] == b"\r\n\r\n":
                break

        result = json.loads(buf.decode("utf-8"))["result"]
        if isinstance(result, str):
            result = json.loads(result)
        return result

    def get_pose(self) -> dict:
        """回傳 {'x': m, 'y': m, 'angle': rad}（內部把 yaw 統一轉成 angle）"""
        raw = self._send("getpose")
        # M2M2 回傳欄位是 yaw，統一映射成 angle 供後續使用
        if "yaw" in raw and "angle" not in raw:
            raw["angle"] = raw["yaw"]
        return raw

    def get_known_area(self) -> dict:
        """回傳 M2M2 已探索範圍 {'min_x', 'max_x', 'min_y', 'max_y'}"""
        return self._send("getknownarea")

    def get_map(self, area: dict | None = None) -> dict:
        """回傳佔用格地圖資料（統一映射成 width/height/resolution/offset_x/offset_y/cells）
        area: {'x': min_x, 'y': min_y, 'width': w, 'height': h} 公尺
        若不指定 area，自動用 getknownarea 取得全圖範圍。
        """
        if area is None:
            ka = self.get_known_area()
            area = {
                "x":      ka.get("min_x", 0.0),
                "y":      ka.get("min_y", 0.0),
                "width":  ka.get("max_x", 0.0) - ka.get("min_x", 0.0),
                "height": ka.get("max_y", 0.0) - ka.get("min_y", 0.0),
            }
        raw = self._send("getmapdata", args={"area": area})
        w = raw.get("dimension_x", 0)
        h = raw.get("dimension_y", 0)
        return {
            "width":      w,
            "height":     h,
            "resolution": area["width"] / w if w > 0 else 0.05,
            "offset_x":   area["x"],
            "offset_y":   area["y"],
            "cells":      raw.get("map_data", ""),
        }

    @staticmethod
    def decompress_rle(b64_data: str) -> bytearray:
        """與 lidar_node 相同的 RLE 解壓縮"""
        rle = base64.b64decode(b64_data)
        if rle[0:3] != b"RLE":
            return bytearray(rle)  # 未壓縮，直接回傳
        sentinel = [rle[3], rle[4]]
        pos = 9
        out: list[int] = []
        while pos < len(rle):
            b = rle[pos]
            if b == sentinel[0]:
                if rle[pos + 1] == 0 and rle[pos + 2] == sentinel[1]:
                    sentinel.reverse()
                    pos += 2
                else:
                    out.extend([rle[pos + 2]] * rle[pos + 1])
                    pos += 2
            else:
                out.append(b)
            pos += 1
        return bytearray(out)


# ─────────────────────────────────────────────────────────────
#  路徑規劃
# ─────────────────────────────────────────────────────────────
def set_custom_exit(px: int, py: int) -> tuple[int, int] | None:
    """
    從 MJPEG 串流圖片的像素座標設定自訂出口點。
    px, py：點擊在串流圖片上的像素座標。
    回傳換算後的格子座標，若超出範圍回傳 None。
    """
    global _custom_exit
    with _custom_exit_lock:
        meta = _map_meta
        scale = meta["scale"]
        w     = meta["w"]
        h     = meta["h"]
        if scale <= 0 or w <= 0 or h <= 0:
            return None
        gx = px // scale
        gy = py // scale
        if not (0 <= gx < w and 0 <= gy < h):
            return None
        _custom_exit = (gx, gy)
        return _custom_exit


def clear_custom_exit():
    """清除自訂出口點，回到自動偵測模式。"""
    global _custom_exit
    with _custom_exit_lock:
        _custom_exit = None


def get_custom_exit() -> tuple[int, int] | None:
    """回傳目前自訂出口格子座標，無則回傳 None。"""
    with _custom_exit_lock:
        return _custom_exit


def set_custom_start(px: int, py: int) -> tuple[int, int] | None:
    """
    從 MJPEG 串流圖片的像素座標設定自訂起點。
    px, py：點擊在串流圖片上的顯示像素座標。
    回傳換算後的格子座標，若超出範圍回傳 None。
    """
    global _custom_start
    with _custom_start_lock:
        meta = _map_meta
        scale = meta["scale"]
        w     = meta["w"]
        h     = meta["h"]
        if scale <= 0 or w <= 0 or h <= 0:
            return None
        gx = px // scale
        gy = py // scale
        if not (0 <= gx < w and 0 <= gy < h):
            return None
        _custom_start = (gx, gy)
        return _custom_start


def clear_custom_start():
    """清除自訂起點，回到使用機器人即時位姿。"""
    global _custom_start
    with _custom_start_lock:
        _custom_start = None


def get_custom_start() -> tuple[int, int] | None:
    """回傳目前自訂起點格子座標，無則回傳 None。"""
    with _custom_start_lock:
        return _custom_start


# ─────────────────────────────────────────────────────────────
#  地圖存檔 / 載入
# ─────────────────────────────────────────────────────────────
def save_map(name: str, map_data: dict, pose: dict) -> str:
    """
    把 M2M2 回傳的 map_data + pose 存成 ~/maps/{name}.json。
    同時記錄目前的自訂出口點與時間戳。
    回傳完整檔案路徑字串。
    """
    MAPS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-") or "map"
    path = MAPS_DIR / f"{safe_name}.json"
    with _custom_exit_lock:
        ce = _custom_exit
    payload = {
        "map":         map_data,
        "pose":        pose,
        "custom_exit": list(ce) if ce else None,
        "map_name":    safe_name,
        "saved_at":    time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False))
    print(f"[SlamViz] 地圖已儲存：{path}")
    return str(path)


def load_map(name: str) -> bool:
    """
    從 ~/maps/{name}.json 載入地圖。
    載入後 _render() 改用離線資料，不再呼叫 M2M2 getmap。
    同時還原 custom_exit。
    回傳是否成功。
    """
    global _offline_map, _custom_exit
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
    path = MAPS_DIR / f"{safe_name}.json"
    if not path.exists():
        print(f"[SlamViz] 找不到地圖檔：{path}")
        return False
    try:
        payload = json.loads(path.read_text())
        with _offline_lock:
            _offline_map = payload
        # 還原自訂出口
        ce = payload.get("custom_exit")
        with _custom_exit_lock:
            _custom_exit = tuple(ce) if ce else None
        print(f"[SlamViz] 地圖已載入（離線模式）：{path}")
        return True
    except Exception as e:
        print(f"[SlamViz] 載入失敗：{e}")
        return False


def unload_map():
    """切回即時 SLAM 模式，清除離線地圖。"""
    global _offline_map
    with _offline_lock:
        _offline_map = None
    print("[SlamViz] 已切回即時 SLAM 模式")


def get_map_list() -> list[dict]:
    """
    回傳 ~/maps/ 下所有 .json 地圖的清單。
    每筆為 {"name": str, "saved_at": str, "size_kb": float}。
    """
    if not MAPS_DIR.exists():
        return []
    result = []
    for p in sorted(MAPS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(p.read_text())
            result.append({
                "name":     p.stem,
                "saved_at": meta.get("saved_at", ""),
                "size_kb":  round(p.stat().st_size / 1024, 1),
            })
        except Exception:
            result.append({"name": p.stem, "saved_at": "", "size_kb": 0})
    return result


def delete_map(name: str) -> bool:
    """刪除 ~/maps/{name}.json，回傳是否成功。"""
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
    path = MAPS_DIR / f"{safe_name}.json"
    if path.exists():
        path.unlink()
        print(f"[SlamViz] 地圖已刪除：{path}")
        return True
    return False


def is_offline_mode() -> bool:
    """是否目前使用離線地圖。"""
    with _offline_lock:
        return _offline_map is not None


def _find_exit(grid: np.ndarray, robot_gx: int, robot_gy: int) -> tuple[int, int] | None:
    """
    在佔用格地圖上找出口：
    掃描前方 ±120° 範圍內，找距離機器人最遠的自由格（非障礙、非未知）
    作為目標出口。
    """
    h, w = grid.shape
    best: tuple[int, int] | None = None
    max_dist = 0.0

    for gy in range(h):
        for gx in range(w):
            if 0 < grid[gy, gx] < 100:  # 已探索且自由
                d = math.hypot(gx - robot_gx, gy - robot_gy)
                if d > max_dist:
                    max_dist = d
                    best = (gx, gy)
    return best


def _astar(
    grid: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]] | None:
    """A* 路徑規劃，回傳格子座標列表（含起點與終點）"""
    h, w = grid.shape

    def heur(a: tuple, b: tuple) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple, tuple] = {}
    g_cost: dict[tuple, float] = {start: 0.0}
    closed: set[tuple[int, int]] = set()
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1),
            (-1, -1), (1, -1), (-1, 1), (1, 1)]

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path: list[tuple[int, int]] = []
            while cur in came_from:
                path.append(cur)
                cur = came_from[cur]
            path.append(start)
            path.reverse()
            return path
        if cur in closed:
            continue
        closed.add(cur)

        for dx, dy in dirs:
            nx, ny = cur[0] + dx, cur[1] + dy
            if not (0 <= nx < w and 0 <= ny < h):
                continue
            if grid[ny, nx] >= 100:  # 障礙格
                continue
            step = 1.414 if (dx and dy) else 1.0
            ng = g_cost[cur] + step
            nb = (nx, ny)
            if ng < g_cost.get(nb, float('inf')):
                g_cost[nb] = ng
                came_from[nb] = cur
                heapq.heappush(open_set, (ng + heur(nb, goal), nb))

    return None  # 找不到路徑


# ─────────────────────────────────────────────────────────────
#  地圖渲染
# ─────────────────────────────────────────────────────────────
def _render(map_data: dict, pose: dict) -> np.ndarray | None:
    """
    把 M2M2 地圖資料渲染成 BGR OpenCV image。
    包含：佔用格著色、機器人位置、方向箭頭、A* 路徑、出口標記。
    """
    try:
        w   = int(map_data.get("width",  0))
        h   = int(map_data.get("height", 0))
        res = float(map_data.get("resolution", 0.05))   # 公尺/格
        ox  = float(map_data.get("offset_x", 0.0))      # 地圖原點 x（公尺）
        oy  = float(map_data.get("offset_y", 0.0))      # 地圖原點 y（公尺）

        if w <= 0 or h <= 0:
            return None

        cells_b64 = map_data.get("cells", "")
        if not cells_b64:
            return None

        raw = SlamMapper.decompress_rle(cells_b64)
        # 確保長度足夠
        raw = bytearray(raw) + bytearray(max(0, w * h - len(raw)))
        arr = np.array(raw[:w * h], dtype=np.uint8).reshape((h, w))

        # 建立 BGR 圖
        # 0   = 未知  → 灰色
        # 1~99 = 自由  → 白色
        # ≥100 = 障礙  → 黑色
        img = np.full((h, w, 3), 160, dtype=np.uint8)   # 預設灰色
        img[arr > 0,  :] = 255                           # 自由 → 白
        img[arr >= 100, :] = 0                           # 障礙 → 黑

        # M2M2 地圖資料 row 0 = 世界座標 y_min（底部），圖片 row 0 卻顯示在畫面頂部
        # → 需垂直翻轉才能正確顯示（北方朝上）
        img = img[::-1, :]

        # ── 放大顯示 ──────────────────────────────────────────
        # 目標短邊 = 480px，限最大 8 倍（4m×4m 區域約 80 cells，scale=6 → 480px）
        scale = max(1, min(8, 480 // min(w, h)))
        disp_w, disp_h = w * scale, h * scale
        disp = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)

        # 更新地圖尺寸快取（供 set_custom_exit 換算座標）
        _map_meta["w"]     = w
        _map_meta["h"]     = h
        _map_meta["scale"] = scale

        # ── 機器人位姿 ───────────────────────────────────────
        robot_gx: int | None = None
        robot_gy: int | None = None

        if pose:
            rx = float(pose.get("x", 0.0))
            ry = float(pose.get("y", 0.0))
            ra = float(pose.get("angle", 0.0))   # rad，地圖座標系

            # 世界座標 → 格子座標（y 軸翻轉：地圖 y 往上，圖片 y 往下）
            robot_gx = int((rx - ox) / res)
            robot_gy = h - 1 - int((ry - oy) / res)

            # 格子座標 → 像素座標（格子中心）
            px = robot_gx * scale + scale // 2
            py = robot_gy * scale + scale // 2

            if 0 <= px < disp_w and 0 <= py < disp_h:
                # 機器人圓點（橘色）
                cv2.circle(disp, (px, py), max(5, scale * 2), (0, 120, 255), -1)
                # 朝向箭頭（綠色）
                alen = scale * 5
                ex = int(px + alen * math.cos(-ra))
                ey = int(py + alen * math.sin(-ra))
                cv2.arrowedLine(disp, (px, py), (ex, ey),
                                (0, 200, 60), 2, tipLength=0.4)

        # ── A* 路徑 ──────────────────────────────────────────
        # 取得自訂起點（若有設定則優先使用，否則用機器人位姿）
        with _custom_start_lock:
            custom_start = _custom_start

        # robot_gx/gy 可能為 None（無 pose），自訂起點也可獨立運作
        astar_start_gx = custom_start[0] if custom_start is not None else robot_gx
        astar_start_gy = custom_start[1] if custom_start is not None else robot_gy

        if astar_start_gx is not None and astar_start_gy is not None:
            # 繪製自訂起點標記（綠色圓形）
            if custom_start is not None:
                cs_px = custom_start[0] * scale + scale // 2
                cs_py = custom_start[1] * scale + scale // 2
                cv2.circle(disp, (cs_px, cs_py), max(5, scale * 2), (0, 200, 60), -1)
                cv2.putText(disp, "START", (cs_px + 6, cs_py - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 60), 1)

            # 在格子座標系（y 已翻轉）做路徑規劃
            # 先建立 flipped grid（y 軸翻轉，對應 disp 方向）
            arr_flip = arr[::-1, :]  # 翻轉 y
            start_pt = (astar_start_gx, astar_start_gy)

            # 優先使用自訂出口點，否則自動偵測
            with _custom_exit_lock:
                custom = _custom_exit
            using_custom = custom is not None
            exit_pt = custom if using_custom else _find_exit(arr_flip, astar_start_gx, astar_start_gy)

            if exit_pt:
                # 出口標記：自訂=黃色星形，自動=紅色星形
                ex_px = exit_pt[0] * scale + scale // 2
                ex_py = exit_pt[1] * scale + scale // 2
                exit_color = (0, 220, 220) if using_custom else (0, 0, 220)
                cv2.drawMarker(disp, (ex_px, ex_py), exit_color,
                               cv2.MARKER_STAR, max(12, scale * 4), 2)
                # 自訂模式標籤
                if using_custom:
                    cv2.putText(disp, "GOAL", (ex_px + 6, ex_py - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)

                # 距離 > 10 格才做 A*（避免起點=終點）
                if math.hypot(exit_pt[0] - astar_start_gx, exit_pt[1] - astar_start_gy) > 10:
                    # 地圖較大時先縮小格子以加快 A*
                    subsample = max(1, min(w, h) // 80)
                    if subsample > 1:
                        small = arr_flip[::subsample, ::subsample]
                        s_start = (astar_start_gx // subsample, astar_start_gy // subsample)
                        s_exit  = (exit_pt[0] // subsample, exit_pt[1] // subsample)
                        path = _astar(small, s_start, s_exit)
                        if path:
                            # 還原為原始格子座標
                            path = [(p[0] * subsample, p[1] * subsample) for p in path]
                    else:
                        path = _astar(arr_flip, start_pt, exit_pt)

                    if path and len(path) > 1:
                        pts = np.array(
                            [[p[0] * scale + scale // 2,
                              p[1] * scale + scale // 2] for p in path],
                            dtype=np.int32
                        )
                        cv2.polylines(disp, [pts], False, (0, 0, 255), 2)

                        # ── 計算導航角度，供 s7_tunnel 實際使用 ──────────
                        # 取路徑上 lookahead 個點之後的目標，避免追太近的點抖動
                        if pose:
                            rx_w  = float(pose.get("x",     0.0))
                            ry_w  = float(pose.get("y",     0.0))
                            ra_w  = float(pose.get("angle", 0.0))
                            look  = min(8, len(path) - 1)
                            nx_g, ny_g = path[look]
                            # 格子座標（y 已翻轉）→ 世界座標（公尺）
                            wx = nx_g * res + ox
                            wy = (h - 1 - ny_g) * res + oy
                            ddx, ddy = wx - rx_w, wy - ry_w
                            if math.hypot(ddx, ddy) > 0.05:
                                tgt_a = math.atan2(ddy, ddx)
                                rel_a = tgt_a - ra_w
                                # 正規化到 [-π, π]
                                rel_a = (rel_a + math.pi) % (2 * math.pi) - math.pi
                                with _nav_lock:
                                    _slam_nav["angle"] = math.degrees(rel_a)
                                    _slam_nav["ts"]    = time.time()

        # ── 資訊文字 ─────────────────────────────────────────
        cv2.putText(disp, "SLAM Map",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 220), 1)
        cv2.putText(disp, f"res={res*100:.1f}cm  {w}x{h}cells",
                    (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (100, 100, 100), 1)

        # 圖例
        legend_y = disp_h - 84
        cv2.circle(disp, (12, legend_y),      5, (0, 120, 255), -1)
        cv2.putText(disp, "Robot",           (22, legend_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 120, 255), 1)
        cv2.circle(disp, (12, legend_y + 16), 5, (0, 200, 60), -1)
        cv2.putText(disp, "Start",           (22, legend_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 60), 1)
        cv2.drawMarker(disp, (12, legend_y + 32), (0, 0, 220),
                       cv2.MARKER_STAR, 10, 1)
        cv2.putText(disp, "Auto Goal",       (22, legend_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 220), 1)
        cv2.drawMarker(disp, (12, legend_y + 48), (0, 220, 220),
                       cv2.MARKER_STAR, 10, 1)
        cv2.putText(disp, "Custom Goal",     (22, legend_y + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)
        cv2.line(disp, (6, legend_y + 62), (18, legend_y + 62), (0, 0, 255), 2)
        cv2.putText(disp, "A* path",         (22, legend_y + 66),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        return disp

    except Exception as e:
        print(f"[SlamViz] render error: {e}")
        return None


# ─────────────────────────────────────────────────────────────
#  背景執行緒
# ─────────────────────────────────────────────────────────────
def _worker():
    global _latest_frame, _running
    mapper = SlamMapper()
    connected = False

    while _running:
        try:
            # ── 離線模式：直接用存好的地圖，不連 M2M2 ──
            with _offline_lock:
                offline = _offline_map

            if offline is not None:
                map_data = offline["map"]
                pose     = offline.get("pose", {})
                img      = _render(map_data, pose)
                if img is not None:
                    ok, buf = cv2.imencode(
                        '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        with _frame_lock:
                            _latest_frame = buf.tobytes()
                time.sleep(1.0)
                continue

            # ── 即時模式：連 M2M2 取得最新地圖 ──
            if not connected:
                mapper.connect()
                connected = True
                print("[SlamViz] 連線 M2M2 成功")

            pose = mapper.get_pose()
            # 只抓機器人周圍 4m x 4m，解析度更高、更新更快
            half = 1.0   # 隧道 2m×2m，只取機器人周圍 2m×2m（解析度比 4m×4m 高一倍）
            rx = float(pose.get("x", 0.0))
            ry = float(pose.get("y", 0.0))
            focus_area = {
                "x":      rx - half,
                "y":      ry - half,
                "width":  half * 2,
                "height": half * 2,
            }
            map_data = mapper.get_map(area=focus_area)
            img      = _render(map_data, pose)

            if img is not None:
                ok, buf = cv2.imencode(
                    '.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    with _frame_lock:
                        _latest_frame = buf.tobytes()

        except Exception as e:
            print(f"[SlamViz] 錯誤: {e}")
            connected = False
            try:
                mapper.disconnect()
            except Exception:
                pass
            time.sleep(2.0)
            continue

        time.sleep(1.0)   # 每秒更新一次地圖

    try:
        mapper.disconnect()
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
#  公開 API
# ─────────────────────────────────────────────────────────────
def save_map_now(name: str) -> str | None:
    """
    即時從 M2M2 抓取地圖並儲存（在即時模式下呼叫）。
    若目前是離線模式，直接把已載入的地圖另存新名稱。
    回傳儲存路徑，失敗回傳 None。
    """
    with _offline_lock:
        offline = _offline_map

    if offline is not None:
        # 離線模式：把目前地圖另存
        return save_map(name, offline["map"], offline.get("pose", {}))

    # 即時模式：重新連線抓最新地圖
    mapper = SlamMapper()
    try:
        mapper.connect()
        map_data = mapper.get_map()
        pose     = mapper.get_pose()
        return save_map(name, map_data, pose)
    except Exception as e:
        print(f"[SlamViz] save_map_now 失敗：{e}")
        return None
    finally:
        mapper.disconnect()


def _restore_exit_from_save(name: str = "tunnel_map") -> bool:
    """
    啟動時從 ~/maps/{name}.json 還原 custom_exit，但不切換離線模式。
    即時 SLAM 會用真實機器人位置 + 還原的出口點跑 A*，不需手動操作。
    """
    global _custom_exit
    safe_name = "".join(c for c in name if c.isalnum() or c in "_-")
    path = MAPS_DIR / f"{safe_name}.json"
    if not path.exists():
        print(f"[SlamViz] 找不到 {path}，出口點未還原（將使用自動偵測）")
        return False
    try:
        payload = json.loads(path.read_text())
        ce = payload.get("custom_exit")
        if ce:
            with _custom_exit_lock:
                _custom_exit = tuple(ce)
            print(f"[SlamViz] 出口點已從 {name}.json 還原：{_custom_exit}（即時 SLAM 模式）")
            return True
        else:
            print(f"[SlamViz] {name}.json 無儲存出口點，將使用自動偵測")
    except Exception as e:
        print(f"[SlamViz] 還原出口點失敗：{e}")
    return False


def start_viz():
    global _worker_thread, _running
    if _worker_thread and _worker_thread.is_alive():
        return
    _running = True
    _worker_thread = threading.Thread(target=_worker, daemon=True, name="SlamViz")
    _worker_thread.start()
    print("[SlamViz] 背景執行緒已啟動")
    # 自動還原上次設定的出口點（不切換離線模式，繼續用即時 SLAM）
    _restore_exit_from_save("tunnel_map")


def stop_viz():
    global _running
    _running = False


def get_nav_angle() -> float | None:
    """
    回傳 A* 路徑導向角度（單位：度）。
      正值 → 左轉（出口在左前方）
      負值 → 右轉（出口在右前方）
      None → 無有效路徑或資料已超過 3 秒未更新
    """
    with _nav_lock:
        if time.time() - _slam_nav.get("ts", 0.0) > 3.0:
            return None
        return _slam_nav.get("angle")


def get_latest_frame() -> bytes | None:
    with _frame_lock:
        return _latest_frame
