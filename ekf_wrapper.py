import numpy as np
from ekf import EKF


class EKFWrapper:
    """
    Wrapper class for the EKF.
    """

    def __init__(self,
                 initial_state: np.ndarray,
                 initial_covariance: np.ndarray,
                 imu_noise: np.ndarray,
                 process_noise_covariance: np.ndarray):
        """
        Initialize the EKF.

        :param initial_state (np.ndarray): The initial state vector.
        :param initial_covariance (np.ndarray): The initial covariance.
        :param imu_noise (np.ndarray): The IMU noise vector (accelerometer x, y, z and gyroscope x, y, z).
        :param process_noise_covariance (np.ndarray): The process noise covariance matrix.
        """
        self.ekf = EKF()
        self.state = initial_state
        self.state_covariance = initial_covariance
        self.process_noise_covariance = process_noise_covariance
        self.imu_noise = imu_noise

    def get_state(self) -> np.ndarray:
        """
        Get the current state.

        :return: The current state vector.
        """
        return self.state

    def get_state_covariance(self) -> np.ndarray:
        """
        Get the current state covariance.

        :return: The current state covariance matrix.
        """
        return self.state_covariance

    def predict(self,
                imu_measurement: np.ndarray,
                dt: float):
        """
        Predict the next state.

        :param imu_measurement (np.ndarray): The IMU measurement vector.
        :param dt (float): The time step.
        """
        X_new, P_new = self.ekf.predict_function(
            self.state,
            imu_measurement,
            self.imu_noise,
            dt,
            self.state_covariance,
            self.process_noise_covariance
        )
        self.state = X_new
        self.state_covariance = P_new

    def update(self,
               z: np.ndarray,
               measurement_noise_covariance: np.ndarray):
        """
        Update the state with a new measurement.

        :param z (np.ndarray): The measurement vector.
        :param measurement_noise_covariance (np.ndarray): The measurement noise covariance matrix.
        """
        X_new, P_new = self.ekf.update_function(
            self.state,
            z,
            self.state_covariance,
            measurement_noise_covariance,
        )
        self.state = X_new
        self.state_covariance = P_new
