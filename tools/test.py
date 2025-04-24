import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from livox_ros_driver2.msg import CustomMsg
import numpy as np
from std_msgs.msg import Header

class LivoxConverter(Node):
    def __init__(self):
        super().__init__('livox_converter')
        self.subscription = self.create_subscription(
            CustomMsg,
            '/livox/lidar', # 
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(PointCloud2, '/kitti/point_cloud', 10)
        self.get_logger().info('LivoxConverter node is running. Waiting for messages...')
        self.frame_number = 0

    def listener_callback(self, msg):
        self.get_logger().info('Received a message. Converting and publishing...')
        points = np.array([(p.x, p.y, -p.z) for p in msg.points], dtype=np.float32)
        points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
        print(f"x min:{np.min(points[:,0])}")
        
        pc2_msg = PointCloud2()
        #pc2_msg.header = msg.header
        
        pc2_msg.header = Header()
        pc2_msg.header.stamp = self.get_clock().now().to_msg()
        pc2_msg.header.frame_id = "livox_frame"

        pc2_msg.height = 1
        pc2_msg.width = len(points)
        pc2_msg.fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1)
        ]
        pc2_msg.is_bigendian = False
        pc2_msg.point_step = 16
        pc2_msg.row_step = pc2_msg.point_step * pc2_msg.width
        pc2_msg.is_dense = True
        pc2_msg.data = points.astype(np.float32).tobytes() # points.tobytes()

        self.publisher.publish(pc2_msg)


        #if self.frame_number == 320:
        #    points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=np.float32)))
        #    np.save(f'./lidar_frame_{self.frame_number}.npy', points)

        self.frame_number += 1

def main(args=None):
    print("here")
    rclpy.init(args=args)
    livox_converter = LivoxConverter()
    try:
        rclpy.spin(livox_converter)
    except KeyboardInterrupt:
        pass
    finally:
        livox_converter.get_logger().info('Shutting down LivoxConverter node...')
        livox_converter.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
