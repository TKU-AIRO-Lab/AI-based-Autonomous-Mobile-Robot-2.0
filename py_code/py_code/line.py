import cv2
import numpy as np
from geometry_msgs.msg import Twist
from .param_server import shared_params

class LineFollowerBase:
    def __init__(self):
        self.base_speed = 0.1
        self.Kp = 0.002
        self.Kd = 0.00015
        self._prev_error_x = 0

    def compute_trapezoid_points(self, image_shape, top_y_ratio, top_width_ratio, bottom_width_ratio, center_x_ratio):
        h, w = image_shape[:2]
        y_bottom = h
        y_top = int(h * top_y_ratio)
        top_width = int(w * top_width_ratio)
        bottom_width = int(w * bottom_width_ratio)
        center_x = int(w * center_x_ratio)
        x_left_top = center_x - top_width // 2
        x_right_top = center_x + top_width // 2
        bottom_center_x = w // 2
        x_left_bottom = bottom_center_x - bottom_width // 2
        x_right_bottom = bottom_center_x + bottom_width // 2

        def clamp_x(x): return max(0, min(x, w - 1))
        return [(clamp_x(x_left_bottom), y_bottom), (clamp_x(x_right_bottom), y_bottom),
                (clamp_x(x_right_top), y_top), (clamp_x(x_left_top), y_top)]

    def process_image(self, cv_image, turn_direction="straight"):
        p = shared_params.get("line", {})
        self.Kp = p.get("Kp", 0.003)
        self.Kd = p.get("Kd", 0.0)
        self.base_speed = p.get("base_speed", 0.05)

        # 影像前處理
        brightness = p.get("brightness", 0)
        sat_scale = p.get("saturation_scale", 1.0)
        if brightness != 0 or sat_scale != 1.0:
            hsv_adj = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv_adj[:,:,1] *= sat_scale
            hsv_adj[:,:,2] += brightness
            hsv_adj = np.clip(hsv_adj, 0, 255).astype(np.uint8)
            cv_image = cv2.cvtColor(hsv_adj, cv2.COLOR_HSV2BGR)

        h, w = cv_image.shape[:2]
        p4 = shared_params.get("s4", {})
        top_y_ratio = p4.get("roi_top_ratio", 0.7) if turn_direction == "nearsighted" else p.get("line_mask_size", 0.5)

        c_ratio = 0.5

        top_w = p.get("line_mask_top_w", 0.5)
        points = self.compute_trapezoid_points(cv_image.shape, top_y_ratio, top_w, 1.0, c_ratio)

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array([points], dtype=np.int32)
        cv2.fillPoly(roi_mask, pts, 255)
        masked_img = cv2.bitwise_and(cv_image, cv_image, mask=roi_mask)

        hsv = cv2.cvtColor(masked_img, cv2.COLOR_BGR2HSV)
        yellow_low  = np.array([p.get("yellow_h_low",20),  p.get("yellow_s_low",100), p.get("yellow_v_low",100)])
        yellow_high = np.array([p.get("yellow_h_high",40), p.get("yellow_s_high",255), p.get("yellow_v_high",255)])
        white_low   = np.array([p.get("white_h_low",0),    p.get("white_s_low",0),    p.get("white_v_low",200)])
        white_high  = np.array([p.get("white_h_high",180), p.get("white_s_high",50),  p.get("white_v_high",255)])
        yellow_mask = cv2.inRange(hsv, yellow_low, yellow_high)
        white_mask  = cv2.inRange(hsv, white_low, white_high)

        if turn_direction == "follow_right":
            yellow_mask[:] = 0
            white_mask[:, : int(w * 0.8)] = 0

        is_aligned = False
        top_y = int(h * top_y_ratio)

        if turn_direction == "spin_left":
            if np.sum(yellow_mask[top_y:h, 0 : w//4] == 255) > 100: is_aligned = True
        elif turn_direction == "spin_right":
            if np.sum(white_mask[top_y:h, 3*w//4 : w] == 255) > 100: is_aligned = True

        center_pts = []
        control_target_x = w // 2
        control_y = int(h * 0.75)
        both_lines_count = 0

        for y in range(h - 10, top_y, -10):
            row_y = yellow_mask[y, :]
            row_w = white_mask[y, :]
            y_coords = np.where(row_y == 255)[0]
            w_coords = np.where(row_w == 255)[0]
            curr_target_x = None
            is_valid_dual_line = False

            if turn_direction == "double_yellow":
                if len(y_coords) > 0:
                    left_y = y_coords[0]
                    right_y = y_coords[-1]
                    if right_y - left_y > w * 0.2:
                        curr_target_x = int((left_y + right_y) / 2)
                        both_lines_count += 1
            else:
                if len(y_coords) > 0 and len(w_coords) > 0:
                    y_center, w_center = np.mean(y_coords), np.mean(w_coords)
                    if (y_center < w_center) and (w_center - y_center > w * 0.2):
                        is_valid_dual_line = True
                        curr_target_x = int((y_center + w_center) / 2)
                        both_lines_count += 1

                if not is_valid_dual_line:
                    # 彎道時縮小單線偏移，避免虛擬目標點偏到外側太遠
                    if turn_direction in ("left", "right"):
                        single_offset = int(w * 0.2)
                    else:
                        single_offset = int(w * 0.4)
                    if len(y_coords) > 0 and len(w_coords) == 0:
                        curr_target_x = int(np.mean(y_coords)) + single_offset
                    elif len(w_coords) > 0 and len(y_coords) == 0:
                        curr_target_x = int(np.mean(w_coords)) - (int(w * 0.15) if turn_direction == "follow_right" else single_offset)

            if curr_target_x is not None:
                center_pts.append((curr_target_x, y))
                if abs(y - control_y) <= 10:
                    control_target_x = curr_target_x

        if turn_direction == "double_yellow" or turn_direction == "straight":
            is_aligned = both_lines_count >= 2

        error_x = control_target_x - (w // 2)
        d_error = error_x - self._prev_error_x
        self._prev_error_x = error_x

        twist_cmd = Twist()
        twist_cmd.linear.x = self.base_speed
        twist_cmd.angular.z = float(-error_x) * self.Kp - float(d_error) * self.Kd

        final_view = masked_img.copy()
        if turn_direction == "double_yellow":
            cv2.putText(final_view, "STAGE 5: DOUBLE YELLOW", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(final_view, f"Aligned: {is_aligned}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if is_aligned else (0, 0, 255), 2)

        cv2.polylines(final_view, [np.array([points], np.int32)], True, (0, 0, 255), 2)
        if len(center_pts) > 0:
            cv2.polylines(final_view, [np.array(center_pts, np.int32)], False, (0, 255, 0), 4)

        # 畫出 control_target_x 的點（青色實心圓）和畫面中心線（白色虛線）
        cv2.circle(final_view, (control_target_x, control_y), 8, (255, 255, 0), -1)
        cv2.circle(final_view, (w // 2, control_y), 5, (255, 255, 255), -1)
        cv2.line(final_view, (w // 2, top_y), (w // 2, h), (255, 255, 255), 1)

        # 顯示 error 數值
        cv2.putText(final_view, f"err: {error_x:+d}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return twist_cmd, final_view, is_aligned
