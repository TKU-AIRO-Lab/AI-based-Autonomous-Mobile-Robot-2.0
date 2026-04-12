# 改動紀錄 2026-04-12

## 1. Stage 2 茫走計時：從「進入第二關就開始」改為「看到 T 確認後才開始」

**檔案**：`py_code/s2_intersection.py`、`py_code/main_controller.py`

### 問題
`process_yolo` 對 `t` 的偵測沒有要求 YOLO 確認（confirmed），只要單幀閃出 `t` 就立刻把 state 切成 `"approaching"` 並開始計時。導致剛進入 Stage 2 時（例如 YOLO 短暫誤判），茫走計時就提前跑完。

### 修改內容

**`s2_intersection.py`**：
```python
# 舊
def process_yolo(self, yolo_detection):
    if detection == 't' and self.state == "normal":
        self.state = "approaching"
        self.t_sign_seen_time = time.time()

# 新
def process_yolo(self, yolo_detection, confirmed=False):
    # 必須 YOLO 確認後才開始茫走計時
    if detection == 't' and self.state == "normal" and confirmed:
        self.state = "approaching"
        self.t_sign_seen_time = time.time()
```

**`main_controller.py`**（`yolo_callback`）：
```python
# 舊
self.s2.process_yolo(detection)

# 新
self.s2.process_yolo(detection, confirmed)
```

---

## 2. UI 盲走時間最大值：8 秒 → 12 秒

**檔案**：`/home/tkuai/Downloads/tuning_panel.html`

```javascript
// 舊
approach_duration: {lb:"盲走時間", min:1.0, max:8.0, step:0.5, u:"秒"}

// 新
approach_duration: {lb:"盲走時間", min:1.0, max:12.0, step:0.5, u:"秒"}
```

---

## 3. 各關卡進場 Log 訊息

**檔案**：`py_code/main_controller.py`

### 規則
每一關「進入」的定義是**看到對應號誌且 YOLO 確認後**才算。Stage 6、7 例外（由前一關自動完成觸發，無獨立進場號誌）。

### 新增的 Log 訊息

| 訊息 | 觸發時機 |
|------|----------|
| `現在進入 Stage 1: 等待綠燈` | 程式啟動（原 `All Ready! Waiting for GREEN LIGHT`） |
| `現在進入 Stage 2 (綠燈確認)` | 綠燈 YOLO 確認（原 `Green Light! Stage 2`） |
| `Stage 2: T 路口號誌確認，開始茫走` | T 號誌 YOLO 確認，茫走計時啟動 |
| `現在進入 Stage 3 (禁止進入確認)` | `no_entry` YOLO 確認 |
| `現在進入 Stage 4 (障礙物確認)` | `obstacle` YOLO 確認 |
| `現在進入 Stage 5 (停車號誌確認)` | `parking` YOLO 確認 |
| `現在進入 Stage 6 (stop確認)` | Stage 5 完成後尋線，`stop` YOLO 確認 |
| `現在進入 Stage 7 (tunnel確認)` | Stage 6 完成後尋線，`tunnel` YOLO 確認 |

---

## 4. Stage 5→6、Stage 6→7 的進場條件修正

**檔案**：`py_code/main_controller.py`

### 問題
- Stage 5 停車完成（`s5.state == "done"`）後，程式立刻切到 cs=6，不等中間的尋線段。
- Stage 6 柵欄通過（`s6.state == "done"`）後，同樣立刻切到 cs=7。
- 實際上兩關之間各有一段尋線，應在看到下一關的號誌後才切換。

### 修改內容

```python
# 舊：s5 done 就立刻進入 Stage 6
elif self.cs == 5:
    self.s5.process_yolo(detection)
    if self.s5.state == "done":
        self.cs = 6

# 新：s5 done 後繼續尋線，確認 stop 才進入 Stage 6
elif self.cs == 5:
    self.s5.process_yolo(detection)
    if self.s5.state == "done" and confirmed and detection == 'stop':
        self.cs = 6
        self.get_logger().info("現在進入 Stage 6 (stop確認)")
```

```python
# 舊：s6 done 就立刻進入 Stage 7
if self.cs == 6:
    if confirmed and detection == 'stop':
        self.s6.process_yolo(detection)
    if self.s6.state == "done":
        self.cs = 7

# 新：s6 done 後繼續尋線，確認 tunnel 才進入 Stage 7
if self.cs == 6:
    if confirmed and detection == 'stop':
        self.s6.process_yolo(detection)
    if self.s6.state == "done" and confirmed and detection == 'tunnel':
        self.cs = 7
        self.get_logger().info("現在進入 Stage 7 (tunnel確認)")
```

### 為什麼不用動 `image_callback`

`s5.get_action()` 和 `s6.get_action()` 在 `done` 狀態都回傳 `"line_follow", "straight"`，所以 cs 還在 5 或 6 時機器人會自動繼續尋線，不需要額外處理。
