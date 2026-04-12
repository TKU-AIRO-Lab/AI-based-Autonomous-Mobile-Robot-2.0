# 改動紀錄 2026-04-11

## 1. 尋線防呆：Last Known Center 記憶 + 信心度降速

**檔案**：`py_code/line.py`

### 問題
彎道時內角線消失，單線 fallback 使用固定 `w * 0.4` offset，在不同曲率的彎道下估計不準，導致車子偏出賽道。

### 修改內容

**新增狀態變數（`__init__`）：**
```python
self._last_center_x = None   # 上一次可靠的中心點
self._lost_frames   = 0      # 連續幾幀沒抓到線
self._LOST_MAX      = 20     # 超過這個幀數就停止信任記憶
```

**單線 fallback 改為加權平均（原本是純固定 offset）：**
```python
# 舊：curr_target_x = int(np.mean(y_coords)) + int(w * 0.4)
# 新：
raw_x = int(np.mean(y_coords)) + int(w * 0.4)
curr_target_x = int(raw_x * 0.4 + self._last_center_x * 0.6)  # 記憶佔 60%
```

**防呆 1 — Last Known Center 記憶更新邏輯：**

| 狀態 | 行為 |
|------|------|
| 雙線穩定（`both_lines_count >= 2`） | 完全信任，更新記憶 |
| 單線（`== 1`） | 記憶慢慢跟上（EMA：舊 80% / 新 20%） |
| 完全沒線，記憶還新鮮（`≤ 20 幀`） | 直接用記憶的中心點繼續導向 |
| 超過 20 幀都沒線 | 放棄記憶，避免舊記憶誤導 |

**防呆 2 — 信心度降速：**

| 信心度 | 速度 |
|--------|------|
| 雙線 | `base_speed`（100%） |
| 單線 | `base_speed × 0.75` |
| 靠記憶 | `base_speed × 0.5` |
| 記憶超時 | `0.0`（停車） |

---

## 2. YOLO 確認機制：從「連續 N 次全同」改為「滑動窗口多數決」

**檔案**：`py_code/main_controller.py`

### 問題
- `YOLO_CONFIRM_COUNT = 10`：要求連續 10 幀都是同一個 label。車子高速行駛或抖動時根本做不到，導致關卡無法切換。
- 號誌放在彎道內角只看得到幾幀，10 次連續完全不可能達到。

### 修改內容

```python
# 舊
YOLO_CONFIRM_COUNT = 10
# → 要求 10 幀全部相同

# 新
YOLO_WINDOW_SIZE   = 8   # 看最近幾幀
YOLO_CONFIRM_THRESH = 5  # 窗口內出現幾次就算確認（62.5%）
```

**`_yolo_confirmed` 邏輯改為計數而非全同：**
```python
# 舊：all(d == label for d in self._yolo_window)
# 新：sum(1 for d in self._yolo_window if d == label) >= YOLO_CONFIRM_THRESH
```

**效果對比：**
```
舊：[no_entry, no_entry, t, no_entry, ×10...] → 永遠不確認
新：最近 8 幀裡 no_entry 出現 5 次 → 確認！
```

---

## 3. 雷達避障三個 Bug 修正

**檔案**：`py_code/main_controller.py`

### Bug 1 — 左右閃避方向都一樣（致命）

```python
# 舊：右側有障礙也往右閃，等於衝進去
elif rfd < danger_zone:
    self.lidar_steer = -avoid_steer   # ← 錯！

# 新：右側有障礙往左閃
elif rfd < danger_zone:
    self.lidar_steer = +avoid_steer   # ← 正確
```

### Bug 2 — 正前方（±20°）完全沒有掃描

```python
# 舊：掃 10°-45°（前左）和 315°-350°（前右），0°±10° 死角

# 新：
if deg <= 20 or deg >= 340:   # ±20° 正前方 → 新增
    front_scan.append(dist)
elif 20 < deg <= 60:          # 前左
    left_scan.append(dist)
elif 300 <= deg < 340:        # 前右
    right_scan.append(dist)

# 正前方有障礙時，往空間較大的那側閃：
self.lidar_steer = avoid_steer if lfd > rfd else -avoid_steer
```

### Bug 3 — `s4.get_action()` 從來沒被呼叫

YOLO 看到障礙物後，`s4` 內部切換到 `state="avoiding"` 並回傳 `follow_right`，但 `image_callback` 完全忽略 `s4`，永遠用 `"nearsighted"` 跑線。

```python
# 舊：
twist_cmd, final_view, _ = self.line_follower.process_image(cv_image, turn_direction="nearsighted")

# 新：
action, value = self.s4.get_action()          # 讓 s4 決定模式
turn_dir = value if action == "line_follow" else "nearsighted"
# 雷達緊急閃避優先，否則交給 s4 的模式
twist_cmd, final_view, _ = self.line_follower.process_image(cv_image, turn_direction=turn_dir)
```

---

## 4. 移除移動相關 Log

**檔案**：`py_code/motor_driver.py`

### 問題
`[Motor] 前進左轉` 等方向變更訊息在循線過程中頻繁輸出，終端機刷版看不到重要訊息。

### 刪除內容
- `_last_direction` 狀態變數
- `_to_direction()` 靜態方法
- `get_logger().info(f'[Motor] {direction}')` 這行

### 保留的 Log（終端機現在只會看到）

| 訊息 | 意義 |
|------|------|
| `All Ready! Waiting for GREEN LIGHT` | 節點啟動 |
| `[YOLO 確認] xxx (5/8)` | 號誌辨識確認 |
| `Green Light! Stage 2` 等 | 關卡切換 |
| `強制跳到Stage X` | 手動切換關卡 |
| `RACE FINISHED!` | 完賽 |
| `error` 訊息 | 硬體異常 |
