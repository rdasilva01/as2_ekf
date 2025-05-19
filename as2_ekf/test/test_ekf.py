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


import unittest

import numpy as np

from as2_ekf.ekf_wrapper import EKFWrapper


class TestEKF(unittest.TestCase):
    def setUp(self):
        # Example parameters
        self.initial_state = np.array([
            0.0, 0.0, 0.0,  # Position (x, y, z)
            0.0, 0.0, 0.0,  # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0,  # Orientation (roll, pitch, yaw)
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        self.initial_state = np.atleast_2d(self.initial_state).T
        # print(self.initial_state)
        self.initial_covariance = np.ones((15, 15)) * 1e-5

        # self.accelerometer_noise_density = 1e-2
        # self.accelerometer_random_walk = 1e-4
        # self.gyroscope_noise_density = 1e-4
        # self.gyroscope_random_walk = 1e-6
        self.accelerometer_noise_density = 0.0
        self.accelerometer_random_walk = 0.0
        self.gyroscope_noise_density = 0.0
        self.gyroscope_random_walk = 0.0

        self.ekf_wrapper = EKFWrapper(
            self.initial_state,
            self.initial_covariance,
            self.accelerometer_noise_density,
            self.gyroscope_noise_density,
            self.accelerometer_random_walk,
            self.gyroscope_random_walk)

    def test_wrapper_init(self):
        self.assertIsNotNone(
            self.ekf_wrapper, "EKFWrapper should be initialized.")
        self.assertEqual(self.ekf_wrapper.state.shape,
                         (15, 1),
                         "State shape should be (15,1).")
        # print(self.ekf_wrapper.state)
        # print(self.initial_state)
        self.assertAlmostEqual(self.ekf_wrapper.state.tolist(),
                               self.initial_state.tolist(),
                               msg="Initial state should match.")
        self.assertEqual(self.ekf_wrapper.state_covariance.shape,
                         (15, 15),
                         "State covariance shape should be (15, 15).")
        self.assertAlmostEqual(self.ekf_wrapper.state_covariance.tolist(),
                               self.initial_covariance.tolist(),
                               msg="Initial covariance should match.")
        self.assertEqual(self.ekf_wrapper.imu_noise.shape,
                         (6,),
                         "IMU noise shape should be (6,).")
        self.assertAlmostEqual(self.ekf_wrapper.imu_noise.tolist(),
                               np.array([self.accelerometer_noise_density, self.accelerometer_noise_density, self.accelerometer_noise_density,
                                         self.gyroscope_noise_density, self.gyroscope_noise_density, self.gyroscope_noise_density]).tolist(),
                               msg="IMU noise should match.")
        self.assertEqual(self.ekf_wrapper.process_noise_covariance.shape,
                         (6,),
                         "Process noise covariance shape should be (6,).")
        self.assertAlmostEqual(self.ekf_wrapper.process_noise_covariance.tolist(),
                               np.array([self.accelerometer_noise_density ** 2, self.accelerometer_noise_density ** 2, self.accelerometer_noise_density ** 2,
                                         self.gyroscope_noise_density ** 2, self.gyroscope_noise_density ** 2, self.gyroscope_noise_density ** 2]).tolist(),
                               msg="Process noise covariance should match.")

    def test_predict_1(self):
        """
        Test EKF with no movement.
        """
        # Reset EKF
        self.ekf_wrapper.reset(self.initial_state,
                               self.initial_covariance)

        # Example IMU Measurement
        imu_measurement = np.array([
            0.0, 0.0, 9.81,  # Accelerometer (ax, ay, az)
            0.0, 0.0, 0.0  # Gyroscope (gx, gy, gz)
        ])

        dt = 0.001
        seconds = 1
        steps = int(seconds / dt)

        # Predict the state using the IMU Measurement
        for _ in range(steps):
            self.ekf_wrapper.predict(imu_measurement, dt)

        # Check the state and Covariance
        self.assertEqual(self.ekf_wrapper.state.shape,
                         (15, 1),
                         "State shape should be (15,1).")
        self.assertEqual(self.ekf_wrapper.state_covariance.shape,
                         (15, 15),
                         "State covariance shape should be (15, 15).")
        # print(self.ekf_wrapper.get_state())
        # print(self.initial_state
        np.testing.assert_almost_equal(self.ekf_wrapper.get_state(),
                                       self.initial_state)
        # self.assertAlmostEqual(self.ekf_wrapper.get_state().tolist(),
        #                        self.initial_state.tolist(),
        #                        msg="State should match initial state.")

    def test_predict_2(self):
        """
        Test EKF with 1m/s² in z movement.
        """
        # Reset EKF
        self.ekf_wrapper.reset(self.initial_state,
                               self.initial_covariance)

        # Example IMU Measurement
        imu_measurement = np.array([
            0.0, 0.0, 10.81,  # Accelerometer (ax, ay, az)
            0.0, 0.0, 0.0  # Gyroscope (gx, gy, gz)
        ])

        dt = 1/200
        seconds = 1
        steps = int(seconds / dt)

        # Predict the state using the IMU Measurement
        for _ in range(steps):
            self.ekf_wrapper.predict(imu_measurement, dt)

        # Check the state and Covariance
        self.assertEqual(self.ekf_wrapper.state.shape,
                         (15, 1),
                         "State shape should be (15,1).")
        self.assertEqual(self.ekf_wrapper.state_covariance.shape,
                         (15, 15),
                         "State covariance shape should be (15, 15).")
        # print(self.ekf_wrapper.get_state())
        # print(self.initial_state)
        # self.assertAlmostEqual(self.ekf_wrapper.get_state().tolist(),
        #                        self.initial_state.tolist(),
        #                        msg="State should not match initial state.")

        position_z = self.initial_state[2, 0] + self.initial_state[5, 0] * seconds + \
            0.5 * (imu_measurement[2] - 9.81) * seconds ** 2
        velocity_z = self.initial_state[5, 0] + \
            (imu_measurement[2] - 9.81) * seconds

        movement = np.array([
            0.0, 0.0, position_z,
            0.0, 0.0, velocity_z,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        movement = np.atleast_2d(movement).T

        # print(self.ekf_wrapper.get_state()[2][0])
        # print((self.initial_state + movement)[2][0])

        np.testing.assert_almost_equal(self.ekf_wrapper.get_state(),
                                       self.initial_state + movement)

    def test_predict_3(self):
        """
        Test EKF with 1m/s² in z and 1m/s² in x.
        """
        # Reset EKF
        self.ekf_wrapper.reset(self.initial_state,
                               self.initial_covariance)

        # Example IMU Measurement
        imu_measurement = np.array([
            1.0, 0.0, 10.81,  # Accelerometer (ax, ay, az)
            0.0, 0.0, 0.0  # Gyroscope (gx, gy, gz)
        ])

        dt = 1/200
        seconds = 1
        steps = int(seconds / dt)

        # Predict the state using the IMU Measurement
        for _ in range(steps):
            self.ekf_wrapper.predict(imu_measurement, dt)

        # Check the state and Covariance
        self.assertEqual(self.ekf_wrapper.state.shape,
                         (15, 1),
                         "State shape should be (15,1).")
        self.assertEqual(self.ekf_wrapper.state_covariance.shape,
                         (15, 15),
                         "State covariance shape should be (15, 15).")
        # print(self.ekf_wrapper.get_state())
        # print(self.initial_state)
        # self.assertAlmostEqual(self.ekf_wrapper.get_state().tolist(),
        #                        self.initial_state.tolist(),
        #                        msg="State should not match initial state.")

        position_z = self.initial_state[2, 0] + self.initial_state[5, 0] * seconds + \
            0.5 * (imu_measurement[2] - 9.81) * seconds ** 2
        velocity_z = self.initial_state[5, 0] + \
            (imu_measurement[2] - 9.81) * seconds

        position_x = self.initial_state[0, 0] + \
            self.initial_state[3, 0] * seconds + \
            0.5 * imu_measurement[0] * seconds ** 2
        velocity_x = self.initial_state[3, 0] + imu_measurement[0] * seconds

        movement = np.array([
            position_x, 0.0, position_z,
            velocity_x, 0.0, velocity_z,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        movement = np.atleast_2d(movement).T

        np.testing.assert_almost_equal(self.ekf_wrapper.get_state(),
                                       self.initial_state + movement)


if __name__ == '__main__':
    unittest.main()
