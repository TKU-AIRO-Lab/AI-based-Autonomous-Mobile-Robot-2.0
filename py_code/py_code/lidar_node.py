import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
import socket
import json
import base64
import math
import struct
import time

# 開源的SlamtecMapper 類別
class SlamtecMapper:
    def __init__(self, host, port):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(10.0) # 加上 timeout 防止卡死
        self.socket.connect((host, port))
        self.request_id = 0

    def disconnect(self):
        self.socket.close()

    def _send_request(self, command, args=None):
        request = {
            "command": command,
            "args": args,
            "request_id": self.request_id
        }
        self.request_id += 1
        data = json.dumps(request)
        
        data_ascii = [ord(character) for character in data]
        data_ascii.extend([10, 13, 10, 13, 10]) # M2M2 需要的換行符號

        self.socket.sendall(bytearray(data_ascii))
        received = b""
        while True:
            response = self.socket.recv(1024)
            if not response:
                raise ConnectionError("Socket connection closed by remote")
            received += response
            if received[-4:] == b"\r\n\r\n":
                break

        received_json = json.loads(received.decode("utf-8"))
        if type(received_json["result"]) == str:
            received_json["result"] = json.loads(received_json["result"])

        return received_json["result"]

    def _decompress_rle(self, b64_encoded):
        rle = base64.b64decode(b64_encoded)
        if rle[0:3] != b"RLE":
            return []
        sentinel_list = [rle[3], rle[4]]

        pos = 9
        decompressed = []
        while pos < len(rle):
            b = rle[pos]
            if b == sentinel_list[0]:
                if rle[pos + 1] == 0 and rle[pos + 2] == sentinel_list[1]:
                    sentinel_list.reverse()
                    pos += 2
                else:
                    more = [rle[pos + 2] for i in range(rle[pos + 1])]
                    decompressed.extend(more)
                    pos += 2
            else:
                decompressed.append(b)
            pos += 1
        return decompressed

    def get_laser_scan(self, valid_only=True):
        response = self._send_request(command="getlaserscan")
        decompressed = bytearray(self._decompress_rle(response["laser_points"]))

        pos = 0
        bytes_per_row = 12
        data = []
        while pos + bytes_per_row <= len(decompressed):
            parts = struct.unpack("f f h h", decompressed[pos:pos + bytes_per_row])
            
            distance = parts[0]
            angle_radian = parts[1]
            
            if distance == 100000.0:
                valid = False
            else:
                valid = True
                
            if valid or not valid_only:
                data.append((angle_radian, distance, valid))
                
            # [Fix Bug]: 只在這裡 + 一次 bytes_per_row
            pos += bytes_per_row 

        return data

# ROS 2 Node 
class M2M2RealNode(Node):
    def __init__(self):
        super().__init__('m2m2_real_node')
        self.publisher_ = self.create_publisher(LaserScan, 'scan', 10)
        
        # 你的光達 IP
        self.lidar_ip = '192.168.11.1'
        self.lidar_port = 1445  
        
        self.mapper = None
        self.connect_to_lidar()

        # M2M2 掃描頻率大概是 8~15Hz，我們設 0.1s (10Hz) 來抓資料
        self.timer = self.create_timer(0.1, self.publish_scan)

    def connect_to_lidar(self):
        self.get_logger().info(f'Connecting to M2M2 via TCP at {self.lidar_ip}:{self.lidar_port}...')
        try:
            self.mapper = SlamtecMapper(self.lidar_ip, self.lidar_port)
            self.get_logger().info('Connection and mapping initialized!')
        except Exception as e:
            self.get_logger().error(f'Failed to connect: {e}')
            self.mapper = None

    def publish_scan(self):
        if not self.mapper:
            return

        try:
            # 抓取真實的光達資料
            scan_data = self.mapper.get_laser_scan(valid_only=True)
            
            if not scan_data:
                return

            # 準備 ROS 2 LaserScan 訊息
            msg = LaserScan()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'laser_frame'
            
            # 設定雷達參數
            msg.angle_min = -math.pi
            msg.angle_max = math.pi
            
            # 假設我們將一圈切成 720 等份 (0.5度的解析度)
            num_readings = 720 
            msg.angle_increment = (2.0 * math.pi) / num_readings
            msg.time_increment = 0.0
            msg.scan_time = 0.1
            msg.range_min = 0.1
            msg.range_max = 40.0
            
            # 初始化全部為無限大 (代表沒掃到東西)
            msg.ranges = [float('inf')] * num_readings
            
            # 將抓到的 (angle, distance) 填入對應的 array index
            for angle, dist, valid in scan_data:
                # 把角度正規化到 -pi ~ pi 之間
                while angle > math.pi: angle -= 2 * math.pi
                while angle < -math.pi: angle += 2 * math.pi
                
                # 計算這個角度對應到陣列裡的哪一個 index
                index = int((angle - msg.angle_min) / msg.angle_increment)
                if 0 <= index < num_readings:
                    msg.ranges[index] = dist

            # 發布！
            self.publisher_.publish(msg)
            self.get_logger().debug(f'Published {len(scan_data)} points.')

        except Exception as e:
            self.get_logger().error(f'Error fetching or publishing data: {e}')
            # 斷線後重置，下次 timer 觸發時重連
            try:
                self.mapper.disconnect()
            except Exception:
                pass
            self.mapper = None
            self.get_logger().info('Attempting to reconnect to LiDAR...')
            self.connect_to_lidar()

def main(args=None):
    rclpy.init(args=args)
    node = M2M2RealNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.mapper:
            node.mapper.disconnect()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()