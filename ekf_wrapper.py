import numpy as np
from ekf import EKF


class EKFWrapper:
    """
    Wrapper class for the EKF.
    """

    def __init__(self,
                 initial_state: np.ndarray,
                 initial_covariance_diagonal: np.ndarray,
                 imu_noise: np.ndarray,
                 process_noise_covariance: np.ndarray):
        """
        Initialize the EKF.

        :param initial_state (np.ndarray): The initial state vector.

        """
        self.ekf = EKF()
        self.state = initial_state
        self.state_covariance = initial_covariance_diagonal
        self.process_noise_covariance = process_noise_covariance
        self.imu_noise = imu_noise

    def predict(self,
                imu_measurement: np.ndarray,
                dt: float):
        """
        Predict the next state.
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
        """
        X_new, P_new = self.ekf.update_function(
            self.state,
            z,
            self.state_covariance,
            measurement_noise_covariance,
        )
        self.state = X_new
        self.state_covariance = P_new


def main():
    print("EKF Test")


if __name__ == "__main__":
    main()
