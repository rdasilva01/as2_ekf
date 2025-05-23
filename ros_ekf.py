import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from ekf_wrapper import EKFWrapper
import numpy as np


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        accelerometer_noise_density = 0.0025624546199207194
        accelerometer_random_walk = 8.055323021637122e-05
        gyroscope_noise_density = 0.00011090831806067944
        gyroscope_random_walk = 2.5135360798417067e-06
        # accelerometer_noise_density = 0.01
        # accelerometer_random_walk = 0.1
        # gyroscope_noise_density = 0.001
        # gyroscope_random_walk = 0.01

        # Example parameters
        initial_state = np.array([
            0.0, 0.0, 0.0,  # Position (x, y, z)
            0.0, 0.0, 0.0,  # Velocity (vx, vy, vz)
            1.0, 0.0, 0.0, 0.0,  # Orientation (quaternion w, x, y, z)
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        initial_covariance = np.ones((16, 16)) * 0.00001
        imu_noise = np.array([
            # Accelerometer noise
            accelerometer_noise_density, accelerometer_noise_density, accelerometer_noise_density,
            # Gyroscope noise
            gyroscope_noise_density, gyroscope_noise_density, gyroscope_noise_density
        ])
        process_noise_covariance_diagonal = np.array([
            # Accelerometer noise covariance
            accelerometer_random_walk, accelerometer_random_walk, accelerometer_random_walk,
            # Gyroscope noise covariance
            gyroscope_random_walk, gyroscope_random_walk, gyroscope_random_walk
        ])

        print("Initial state:", initial_state)
        print("Initial covariance diagonal:", initial_covariance)
        print("IMU noise:", imu_noise)
        print("Process noise covariance:", process_noise_covariance_diagonal)

        self.ekf_wrapper = EKFWrapper(
            initial_state,
            initial_covariance,
            imu_noise,
            process_noise_covariance_diagonal)

        print("EKF wrapper initialized.")

        qos_profile = QoSProfile(
            depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/drone0/sensor_measurements/imu',
            self.imu_callback,
            qos_profile
        )
        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.pose_callback,
            10
        )
        self.odom_publisher = self.create_publisher(
            Odometry,
            '/odom',
            10
        )

        self.last_time = 0.0
        self.current_time = 0.001
        self.imu_counter = 0

    def imu_callback(self, msg):
        # Process IMU data

        imu_measurement = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        if self.imu_counter != 0:
            self.current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = self.current_time - self.last_time
        print("dt:", dt)

        self.ekf_wrapper.predict(imu_measurement, dt)
        self.last_time = self.current_time
        self.imu_counter += 1

        # Publish the odometry message
        state = self.ekf_wrapper.get_state()
        covariance = self.ekf_wrapper.get_state_covariance()
        print("State after prediction:", state)
        print("State covariance after prediction:", covariance)
        odom_msg = Odometry()
        odom_msg.header.stamp = msg.header.stamp
        odom_msg.header.frame_id = 'earth'
        odom_msg.child_frame_id = 'base_link'
        odom_msg.pose.pose.position.x = float(state[0])
        odom_msg.pose.pose.position.y = float(state[1])
        odom_msg.pose.pose.position.z = float(state[2])
        quaternion = np.array([
            state[6], state[7], state[8], state[9]])
        quaternion = quaternion / np.linalg.norm(quaternion)
        odom_msg.pose.pose.orientation.w = float(quaternion[0])
        odom_msg.pose.pose.orientation.x = float(quaternion[1])
        odom_msg.pose.pose.orientation.y = float(quaternion[2])
        odom_msg.pose.pose.orientation.z = float(quaternion[3])
        diag_covariance = np.double(np.diag(
            np.append(np.diag(covariance)[0:3], np.diag(covariance)[7:10]))).flatten().tolist()
        print("Covariance diagonal:", diag_covariance)
        odom_msg.pose.covariance = diag_covariance
        odom_msg.twist.twist.linear.x = float(state[3])
        odom_msg.twist.twist.linear.y = float(state[4])
        odom_msg.twist.twist.linear.z = float(state[5])
        odom_msg.twist.twist.angular.x = msg.angular_velocity.x
        odom_msg.twist.twist.angular.y = msg.angular_velocity.y
        odom_msg.twist.twist.angular.z = msg.angular_velocity.z
        odom_msg.twist.covariance = np.double(np.eye(6)).flatten().tolist()
        print("Odometry covariance:")
        print(odom_msg.twist.covariance)
        self.odom_publisher.publish(odom_msg)

    def pose_callback(self, msg):
        # Process initial pose data
        pass


def main(args=None):
    rclpy.init(args=args)

    ekf_node = EKFNode()

    try:
        rclpy.spin(ekf_node)
    except KeyboardInterrupt:
        pass
    ekf_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
