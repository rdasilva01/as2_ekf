import numpy as np
from ekf_wrapper import EKFWrapper


def main():
    print("Test for the EKF wrapper.")

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
        0.0, 0.0, 0.0,  # Orientation (roll, pitch, yaw)
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0
    ])
    initial_covariance = np.ones((15, 15)) * 1e-5

    print("Initial state:", initial_state)
    print("Initial covariance diagonal:", initial_covariance)

    ekf_wrapper = EKFWrapper(
        initial_state,
        initial_covariance,
        accelerometer_noise_density,
        gyroscope_noise_density,
        accelerometer_random_walk,
        gyroscope_random_walk)

    print("EKF wrapper initialized.")

    # Example IMU Measurement
    imu_measurement = np.array([
        0.0, 0.0, 10.81,  # Accelerometer (ax, ay, az)
        0.0, 0.0, 0.0  # Gyroscope (gx, gy, gz)
    ])
    dt = 0.001

    seconds = 1
    steps = int(seconds / dt)

    # Predict the state using the IMU Measurement
    for _ in range(steps):
        ekf_wrapper.predict(imu_measurement, dt)

    print("State after prediction:", ekf_wrapper.get_state())
    print("State covariance after prediction:",
          ekf_wrapper.get_state_covariance())

    # # Example pose Measurement
    # pose_measurement = np.array([
    #     0.0, 0.0, 0.0,  # Position (x, y, z)
    #     0.0, 0.0, 0.0  # Orientation (roll, pitch, yaw)
    # ])
    # pose_covariance = np.ones(6) * 0.01

    # res = ekf_wrapper.update(pose_measurement, pose_covariance)

    # print("State after update:", ekf_wrapper.get_state())
    # print("State covariance after update:",
    #       ekf_wrapper.get_state_covariance())
    # print(res)


if __name__ == "__main__":
    main()
