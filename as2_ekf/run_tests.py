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


import numpy as np
from as2_ekf.ekf_wrapper import EKFWrapper


def main():
    print("Test for the EKF wrapper.")

    # accelerometer_noise_density = 0.0025624546199207194
    # accelerometer_random_walk = 8.055323021637122e-05
    # gyroscope_noise_density = 0.00011090831806067944
    # gyroscope_random_walk = 2.5135360798417067e-06
    accelerometer_noise_density = 1e-2
    accelerometer_random_walk = 1e-4
    gyroscope_noise_density = 1e-4
    gyroscope_random_walk = 1e-6

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
    return ekf_wrapper

    # # Example IMU Measurement
    # imu_measurement = np.array([
    #     0.0, 0.0, 9.81,  # Accelerometer (ax, ay, az)
    #     0.0, 0.0, 0.0  # Gyroscope (gx, gy, gz)
    # ])
    # dt = 0.001

    # seconds = 1
    # steps = int(seconds / dt)

    # # Predict the state using the IMU Measurement
    # for _ in range(steps):
    #     ekf_wrapper.predict(imu_measurement, dt)

    # print("State after prediction:", ekf_wrapper.get_state())
    # print("State covariance after prediction:",
    #       ekf_wrapper.get_state_covariance())

    # # # Example pose Measurement
    # # pose_measurement = np.array([
    # #     0.0, 0.0, 0.0,  # Position (x, y, z)
    # #     0.0, 0.0, 0.0  # Orientation (roll, pitch, yaw)
    # # ])
    # # pose_covariance = np.ones(6) * 0.01

    # # res = ekf_wrapper.update(pose_measurement, pose_covariance)

    # # print("State after update:", ekf_wrapper.get_state())
    # # print("State covariance after update:",
    # #       ekf_wrapper.get_state_covariance())
    # # print(res)


if __name__ == "__main__":
    wrapper = main()
    wrapper.ekf.predict_function.generate('ekf_predict.c')
    wrapper.ekf.update_function.generate('ekf_update.c')
