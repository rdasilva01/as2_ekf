#!/usr/bin/env python3

# Copyright 2025 Universidad Politécnica de Madrid
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""EKF definition."""

__authors__ = 'Rodrigo da Silva Gómez'
__copyright__ = 'Copyright (c) 2025 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'


import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from ekf_wrapper import EKFWrapper
import numpy as np
# import tf_transformations as tf


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    """
    Normalize a quaternion.

    :param q: The quaternion to normalize, shape (4,), [qw, qx, qy, qz].
    :type q: np.ndarray
    :return: The normalized quaternion.
    :rtype: np.ndarray
    """
    norm = np.linalg.norm(q)
    if norm == 0:
        norm = 1e-6
    return q / norm


def quaternion_to_euler(q: np.ndarray) -> np.ndarray:
    """
    Convert a quaternion to Euler angles (roll, pitch, yaw).

    :param q: The quaternion [qw, qx, qy, qz], shape (4,).
    :type q: np.ndarray
    :return: The Euler angles [roll, pitch, yaw], shape (3,).
    :rtype: np.ndarray
    """
    # First, normalize the quaternion to ensure a valid rotation
    q_normed = normalize_quaternion(q)
    qw, qx, qy, qz = q_normed

    # --- Roll (x-axis rotation) ---
    # sinr_cosp = 2 * (qw * qx + qy * qz)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    # cosr_cosp = 1 - 2 * (qx^2 + qy^2)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # --- Pitch (y-axis rotation) ---
    # sinp = 2 * (qw * qy - qz * qx)
    sinp = 2.0 * (qw * qy - qz * qx)
    # Clamp sinp to [-1, 1] to avoid invalid domain for arcsin
    sinp_clamped = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp_clamped)

    # --- Yaw (z-axis rotation) ---
    # siny_cosp = 2 * (qw * qz + qx * qy)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    # cosy_cosp = 1 - 2 * (qy^2 + qz^2)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])


def euler_to_quaternion(euler_angles: np.ndarray) -> np.ndarray:
    """
    Convert Euler angles (roll, pitch, yaw) to a quaternion.

    :param euler_angles: The Euler angles [roll, pitch, yaw], shape (3,).
    :type euler_angles: np.ndarray
    :return: The quaternion [qw, qx, qy, qz], shape (4,).
    :rtype: np.ndarray
    """
    roll, pitch, yaw = euler_angles

    # Compute half angles
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    # Compute quaternion components
    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return np.array([qw, qx, qy, qz])


def compute_tilt_and_euler_angles(g: np.ndarray,
                                  g_ref: np.ndarray = np.array([0., 0., 9.81])) -> tuple:
    """
    Compute the total tilt angle (from vertical) and the roll & pitch Euler angles
    from a measured gravity vector `g` (numpy array of shape (3,)).

    Parameters
    ----------
    g : np.ndarray
        Measured gravity vector [gx, gy, gz] in m/s².
    g_ref : np.ndarray, optional
        Reference gravity vector (default [0, 0, 9.81]).

    Returns
    -------
    tilt : float
        Total tilt angle from vertical (radians).
    roll : float
        Rotation about the x-axis (radians).
    pitch : float
        Rotation about the y-axis (radians).
    """
    # Normalize vectors
    g_norm = np.linalg.norm(g)
    g_unit = g / g_norm
    g_ref_unit = g_ref / np.linalg.norm(g_ref)

    # Total tilt via dot product
    # Clip the dot product for numerical stability
    cos_theta = np.clip(np.dot(g_unit, g_ref_unit), -1.0, 1.0)
    tilt = np.arccos(cos_theta)

    # Euler angles (roll, pitch)
    gx, gy, gz = g
    roll = np.arctan2(gy, gz)
    pitch = np.arctan2(-gx, np.sqrt(gy**2 + gz**2))

    return tilt, roll, pitch


class EKFNode(Node):
    def __init__(self):
        super().__init__('ekf_node')

        accelerometer_noise_density = 0.0025624546199207194
        accelerometer_random_walk = 8.055323021637122e-05
        gyroscope_noise_density = 0.00011090831806067944
        gyroscope_random_walk = 2.5135360798417067e-06
        # accelerometer_noise_density = 0.0
        # accelerometer_random_walk = 0.0
        # gyroscope_noise_density = 0.0
        # gyroscope_random_walk = 0.0
        # accelerometer_noise_density = 1e-1
        # accelerometer_random_walk = 1e-2
        # gyroscope_noise_density = 1e-2
        # gyroscope_random_walk = 1e-3

        # Example parameters
        self.initial_state = np.array([
            0.0, 0.0, 0.0,  # Position (x, y, z)
            0.0, 0.0, 0.0,  # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0,  # Orientation (roll, pitch, yaw)
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        self.initial_covariance = np.ones((15, 15)) * 0.0
        # self.initial_covariance[6:9, 6:9] = np.identity(
        #     3) * 1e-6  # Orientation covariance

        print("Initial state:", self.initial_state)
        print("Initial covariance diagonal:", self.initial_covariance)

        self.ekf_wrapper = EKFWrapper(
            self.initial_state,
            self.initial_covariance,
            accelerometer_noise_density,
            gyroscope_noise_density,
            accelerometer_random_walk,
            gyroscope_random_walk)

        print("EKF wrapper initialized.")

        qos_profile = QoSProfile(
            depth=10, reliability=rclpy.qos.ReliabilityPolicy.BEST_EFFORT)

        self.imu_subscriber = self.create_subscription(
            Imu,
            '/drone0/sensor_measurements/imu',
            self.imu_callback,
            qos_profile
        )
        # self.pose_subscriber = self.create_subscription(
        #     PoseStamped,
        #     '/drone0/self_localization/pose',
        #     self.pose_callback,
        #     qos_profile
        # )
        self.odom_publisher = self.create_publisher(
            Odometry,
            '/ekf_odom',
            10
        )

        self.last_time = 0.0
        self.current_time = 0.0
        self.imu_counter = 0
        self.pose_counter = 0

        self.imu_start = np.array([0.0, 0.0, 0.0])

    def imu_callback(self, msg):
        # print("IMU callback")
        # Process IMU data

        imu_linear_x = msg.linear_acceleration.x
        imu_linear_y = msg.linear_acceleration.y
        imu_linear_z = msg.linear_acceleration.z
        imu_angular_x = msg.angular_velocity.x
        imu_angular_y = msg.angular_velocity.y
        imu_angular_z = msg.angular_velocity.z

        imu_measurement = np.array([
            imu_linear_x,
            imu_linear_y,
            imu_linear_z,
            imu_angular_x,
            imu_angular_y,
            imu_angular_z,
        ])

        self.last_time = self.current_time
        self.current_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = self.current_time - self.last_time
        if self.imu_counter == 0:
            dt = 1/200
        # print("dt:", dt)

        if self.imu_counter > 200 * 5:
            self.ekf_wrapper.predict(imu_measurement, dt)
            self.imu_counter += 1

            # Publish the odometry message
            state = self.ekf_wrapper.get_state().T[0]
            covariance = self.ekf_wrapper.get_state_covariance()
            # print("State after prediction:\n", state)
            # print("State covariance after prediction:", covariance)
            odom_msg = Odometry()
            odom_msg.header.stamp = msg.header.stamp
            odom_msg.header.frame_id = 'earth'
            odom_msg.child_frame_id = 'base_link'
            odom_msg.pose.pose.position.x = float(state[0])
            odom_msg.pose.pose.position.y = float(state[1])
            odom_msg.pose.pose.position.z = float(state[2])
            # print("Position:", odom_msg.pose.pose.position)
            quaternion = euler_to_quaternion(
                np.array([state[6], state[7], state[8]]))
            odom_msg.pose.pose.orientation.w = float(quaternion[0])
            odom_msg.pose.pose.orientation.x = float(quaternion[1])
            odom_msg.pose.pose.orientation.y = float(quaternion[2])
            odom_msg.pose.pose.orientation.z = float(quaternion[3])
            diag_covariance = np.double(np.diag(
                np.append(np.diag(covariance)[0:3], np.diag(covariance)[6:9]))).flatten().tolist()
            # print("Covariance diagonal:", diag_covariance)
            odom_msg.pose.covariance = diag_covariance
            odom_msg.twist.twist.linear.x = float(state[3])
            odom_msg.twist.twist.linear.y = float(state[4])
            odom_msg.twist.twist.linear.z = float(state[5])
            odom_msg.twist.twist.angular.x = msg.angular_velocity.x
            odom_msg.twist.twist.angular.y = msg.angular_velocity.y
            odom_msg.twist.twist.angular.z = msg.angular_velocity.z
            odom_msg.twist.covariance = np.double(np.eye(6)).flatten().tolist()
            # print("Odometry covariance:")
            # print(odom_msg.twist.covariance)
            self.odom_publisher.publish(odom_msg)

            # if self.imu_counter == 1010:
            #     exit()
        elif self.imu_counter == 200 * 5:
            self.imu_start /= self.imu_counter
            print("IMU start:", self.imu_start)
            self.ekf_wrapper.gravity = self.imu_start
            # _, roll, pitch = compute_tilt_and_euler_angles(self.imu_start)
            # self.ekf_wrapper.state[6] = roll
            # self.ekf_wrapper.state[7] = pitch
            # print("Initial roll:", roll)
            # print("Initial pitch:", pitch)
            self.imu_counter += 1
        else:
            # print("IMU counter:", self.imu_counter)
            self.imu_start += np.array([
                imu_linear_x,
                imu_linear_y,
                imu_linear_z,
            ])
            self.imu_counter += 1

    def pose_callback(self, msg):
        if self.pose_counter % (100/5) == 0 and self.imu_counter > 1000:
            print(f"Update {self.imu_counter}")
            # Process initial pose data
            pose_x = msg.pose.position.x
            pose_y = msg.pose.position.y
            pose_z = msg.pose.position.z
            orientation_x = msg.pose.orientation.x
            orientation_y = msg.pose.orientation.y
            orientation_z = msg.pose.orientation.z
            orientation_w = msg.pose.orientation.w
            orientation = quaternion_to_euler(
                np.array([orientation_w, orientation_x, orientation_y, orientation_z]))
            print("Pose orientation (roll, pitch, yaw):", orientation)

            pose_measurement = np.array([
                pose_x, pose_y, pose_z,
                orientation[0], orientation[1], orientation[2],
            ])

            self.ekf_wrapper.update_pose(pose_measurement, np.ones(6) * 1e-3)
            # exit()
        self.pose_counter += 1


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
