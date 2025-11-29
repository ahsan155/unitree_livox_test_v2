# ros_bridge.py   (run with system python that has rclpy, e.g. python3 system)
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import json
import socket
import threading
import time

PRED_SERVER_HOST = '127.0.0.1'
PRED_SERVER_PORT = 6000

# Simple newline-delimited JSON over TCP
def send_json(sock, obj):
    data = json.dumps(obj) + "\n"
    sock.sendall(data.encode('utf-8'))

def recv_lines(sock, callback):
    buf = b""
    while True:
        data = sock.recv(4096)
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            try:
                obj = json.loads(line.decode('utf-8'))
            except Exception as e:
                print("Bad json line:", e)
                continue
            callback(obj)

class RosBridge(Node):
    def __init__(self):
        super().__init__('ros_bridge')
        self.sub = self.create_subscription(String, '/agent_histories', self.on_history, 10)
        self.pred_pub = self.create_publisher(MarkerArray, '/predicted_trajectories', 10)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.settimeout(5.0)
        # connect to prediction server (retry until available)
        connected = False
        for i in range(10):
            try:
                self.socket.connect((PRED_SERVER_HOST, PRED_SERVER_PORT))
                connected = True
                break
            except Exception:
                time.sleep(0.5)
        if not connected:
            self.get_logger().error(f"Could not connect to prediction server at {PRED_SERVER_HOST}:{PRED_SERVER_PORT}")
            raise RuntimeError("Cannot connect to prediction server")

        # start listening thread for incoming prediction messages
        t = threading.Thread(target=recv_lines, args=(self.socket, self.on_pred_msg), daemon=True)
        t.start()
        self.get_logger().info("ROS bridge started and connected to prediction server.")

    def on_history(self, msg: String):
        # Forward history JSON to prediction server directly
        try:
            payload = json.loads(msg.data)
        except Exception as e:
            self.get_logger().error(f"Invalid JSON on /agent_histories: {e}")
            return
        # Send payload to TF server
        send_json(self.socket, {"type": "history", "payload": payload})

    def on_pred_msg(self, obj):
        # Called from socket thread when prediction received
        try:
            if obj.get("type") != "prediction":
                return
            
            # --- CLEAR OLD MARKERS FIRST ---
            clear_msg = Marker()
            clear_msg.action = Marker.DELETEALL
            self.pred_pub.publish(MarkerArray(markers=[clear_msg]))
            
            preds = obj.get("payload", {}).get("predictions", [])
            marker_array = MarkerArray()
            now = self.get_clock().now().to_msg()
            for entry in preds:
                track_id = int(entry.get("track_id", 0))
                traj = entry.get("trajectory", [])  # list of [x,y]
                m = Marker()
                m.header.frame_id = 'map'   # adapt if needed
                m.header.stamp = now
                m.ns = 'predictions'
                m.id = track_id + 1000
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.03
                m.color.r = 1.0
                m.color.g = 0.0
                m.color.b = 0.0
                m.color.a = 0.8
                pts = []
                for p in traj:
                    pt = Point()
                    pt.x = float(p[0])
                    pt.y = float(p[1])
                    pt.z = 0.0
                    pts.append(pt)
                m.points = pts
                marker_array.markers.append(m)
            if marker_array.markers:
                self.pred_pub.publish(marker_array)
        except Exception as e:
            self.get_logger().error(f"Error processing prediction message: {e}")

def main():
    rclpy.init()
    node = RosBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        node.socket.close()

if __name__ == "__main__":
    main()

