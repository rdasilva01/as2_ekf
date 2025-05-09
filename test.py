import numpy as np
from ekf_wrapper import EKFWrapper


def main():
    print("Test for the EKF wrapper.")

    # Example parameters
    initial_state = np.array([
        0.0, 0.0, 0.0,  # Position (x, y, z)
        0.0, 0.0, 0.0,  # Velocity (vx, vy, vz)
        1.0, 0.0, 0.0, 0.0,  # Orientation (quaternion w, x, y, z)
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])
    initial_covariance = np.zeros((16, 16))
    imu_noise = np.array([
        0.0, 0.0, 0.0,    # Accelerometer noise
        0.0, 0.0, 0.0  # Gyroscope noise
    ])
    process_noise_covariance_diagonal = np.array([
        0.0, 0.0, 0.0,     # Accelerometer noise covariance
        0.0, 0.0, 0.0,  # Gyroscope noise covariance
    ])

    print("Initial state:", initial_state)
    print("Initial covariance diagonal:", initial_covariance)
    print("IMU noise:", imu_noise)
    print("Process noise covariance:", process_noise_covariance_diagonal)

    ekf_wrapper = EKFWrapper(
        initial_state,
        initial_covariance,
        imu_noise,
        process_noise_covariance_diagonal)

    print("EKF wrapper initialized.")

    # Example IMU Measurement
    imu_measurement = np.array([
        0.0, 0.0, 10.81,  # Accelerometer (ax, ay, az)
        0.0, 0.0, 0.0  # Gyroscope (gx, gy, gz)
    ])
    dt = 0.01
    ekf_wrapper.predict(imu_measurement, dt)
    ekf_wrapper.predict(imu_measurement, dt)
    ekf_wrapper.predict(imu_measurement, dt)
    ekf_wrapper.predict(imu_measurement, dt)
    ekf_wrapper.predict(imu_measurement, dt)

    print("State after prediction:", ekf_wrapper.get_state())
    print("State covariance after prediction:",
          ekf_wrapper.get_state_covariance())

    # Example pose Measurement
    pose_measurement = np.array([
        0.0, 0.0, 0.0,  # Position (x, y, z)
        1.0, 0.0, 0.0, 0.0  # Orientation (quaternion w, x, y, z)
    ])
    pose_covariance = np.ones(7) * 0.000001

    res = ekf_wrapper.update(pose_measurement, pose_covariance)

    print("State after update:", ekf_wrapper.get_state())
    print("State covariance after update:",
          ekf_wrapper.get_state_covariance())
    print(res)


if __name__ == "__main__":
    main()
