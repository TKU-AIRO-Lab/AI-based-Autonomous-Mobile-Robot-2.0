"""
直接用 dynamixel_sdk 控制 TurtleBot3 Burger 馬達
Motors: XL430-W250, ID 1 (left) & ID 2 (right)
透過 OpenCR USB (/dev/ttyACM0) 以 1 Mbps Dynamixel Protocol 2.0 溝通
"""
import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from dynamixel_sdk import PortHandler, PacketHandler

# XL430-W250 Control Table
ADDR_OPERATING_MODE = 11   # 1 byte
ADDR_TORQUE_ENABLE  = 64   # 1 byte
ADDR_GOAL_VELOCITY  = 104  # 4 bytes (int32)

PROTOCOL_VERSION = 2.0
BAUDRATE         = 1000000
DEVICE           = '/dev/ttyACM0'

DXL_LEFT  = 1
DXL_RIGHT = 2

# TurtleBot3 Burger kinematics
WHEEL_RADIUS     = 0.033   # m
WHEEL_SEPARATION = 0.160   # m

# XL430: 0.229 rpm/unit  →  1 rad/s = 60/(2π*0.229) units
RAD_S_TO_UNIT = 60.0 / (2.0 * math.pi * 0.229)


def to_uint32(val: int) -> int:
    """把有號 int 轉成 Dynamixel 要的 uint32（two's complement）"""
    return val & 0xFFFFFFFF


class MotorDriverNode(Node):
    def __init__(self):
        super().__init__('motor_driver')

        self.ph = PortHandler(DEVICE)
        self.pk = PacketHandler(PROTOCOL_VERSION)
        self.port_ready = False

        # 先建立 subscription，不論硬體是否初始化成功
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_cb, 10)

        if not self.ph.openPort():
            self.get_logger().error(f'Cannot open {DEVICE}')
            return
        if not self.ph.setBaudRate(BAUDRATE):
            self.get_logger().error('Cannot set baud rate to 1 Mbps')
            return

        # 等 OpenCR 準備好
        time.sleep(0.5)

        # 設定 Velocity Control Mode 並開啟 Torque
        init_ok = True
        for dxl_id in (DXL_LEFT, DXL_RIGHT):
            # 1. 先關 Torque（才能改 Operating Mode）
            comm, _ = self.pk.write1ByteTxRx(self.ph, dxl_id, ADDR_TORQUE_ENABLE, 0)
            if comm != 0:
                self.get_logger().error(f'[Motor {dxl_id}] Disable torque failed (code={comm}). Check OpenCR usb_to_dxl firmware.')
                init_ok = False
                continue
            # 2. 設 Velocity Control Mode
            comm, _ = self.pk.write1ByteTxRx(self.ph, dxl_id, ADDR_OPERATING_MODE, 1)
            if comm != 0:
                self.get_logger().error(f'[Motor {dxl_id}] Set velocity mode failed (code={comm})')
                init_ok = False
                continue
            # 3. 開 Torque
            comm, _ = self.pk.write1ByteTxRx(self.ph, dxl_id, ADDR_TORQUE_ENABLE, 1)
            if comm != 0:
                self.get_logger().error(f'[Motor {dxl_id}] Enable torque failed (code={comm})')
                init_ok = False
                continue
            self.get_logger().info(f'Motor {dxl_id} init OK (Velocity Mode, Torque ON)')

        if not init_ok:
            self.get_logger().error('Motor init failed. Motors will not move.')
            return

        self.port_ready = True
        self.get_logger().info('Motor driver ready (ID 1=left, ID 2=right). Port ready.')
        self._startup_vibration()

    def _startup_vibration(self):
        """啟動成功時讓馬達短暫振動，表示連接正常"""
        VIBRATE_UNIT = 50   # 小速度值，約 0.11 rad/s
        PULSE_SEC    = 0.12 # 每次脈衝持續時間

        for _ in range(3):
            # 正轉
            self.pk.write4ByteTxRx(self.ph, DXL_LEFT,  ADDR_GOAL_VELOCITY, to_uint32( VIBRATE_UNIT))
            self.pk.write4ByteTxRx(self.ph, DXL_RIGHT, ADDR_GOAL_VELOCITY, to_uint32(-VIBRATE_UNIT))
            time.sleep(PULSE_SEC)
            # 反轉
            self.pk.write4ByteTxRx(self.ph, DXL_LEFT,  ADDR_GOAL_VELOCITY, to_uint32(-VIBRATE_UNIT))
            self.pk.write4ByteTxRx(self.ph, DXL_RIGHT, ADDR_GOAL_VELOCITY, to_uint32( VIBRATE_UNIT))
            time.sleep(PULSE_SEC)

        # 停止
        self.pk.write4ByteTxRx(self.ph, DXL_LEFT,  ADDR_GOAL_VELOCITY, 0)
        self.pk.write4ByteTxRx(self.ph, DXL_RIGHT, ADDR_GOAL_VELOCITY, 0)
        self.get_logger().info('Startup vibration done.')

    def _cmd_vel_cb(self, msg: Twist):
        if not self.port_ready:
            self.get_logger().error('Port not ready! 請確認 /dev/ttyACM0 是否連接正確', throttle_duration_sec=2.0)
            return

        lx = msg.linear.x
        az = msg.angular.z

        # 輪緣線速度 (m/s) → 角速度 (rad/s) → Dynamixel 單位
        v_left  = (lx - az * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS
        v_right = (lx + az * WHEEL_SEPARATION / 2.0) / WHEEL_RADIUS

        unit_left  = int(v_left  * RAD_S_TO_UNIT)
        unit_right = int(-v_right * RAD_S_TO_UNIT)   # 右馬達方向相反

        self.pk.write4ByteTxRx(self.ph, DXL_LEFT,  ADDR_GOAL_VELOCITY, to_uint32(unit_left))
        self.pk.write4ByteTxRx(self.ph, DXL_RIGHT, ADDR_GOAL_VELOCITY, to_uint32(unit_right))

    def destroy_node(self):
        # 停止馬達並關閉 Torque
        for dxl_id in (DXL_LEFT, DXL_RIGHT):
            self.pk.write4ByteTxRx(self.ph, dxl_id, ADDR_GOAL_VELOCITY, 0)
            self.pk.write1ByteTxRx(self.ph, dxl_id, ADDR_TORQUE_ENABLE, 0)
        self.ph.closePort()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MotorDriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
