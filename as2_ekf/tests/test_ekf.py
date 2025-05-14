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
__copyright__ = 'Copyright (c) 2022 Universidad Politécnica de Madrid'
__license__ = 'BSD-3-Clause'


import unittest

import numpy as np

from as2_ekf.ekf_wrapper import EKFWrapper


class TestEKF(unittest.TestCase):
    def test_wrapper_init(self):
        # Example parameters
        self.initial_state = np.array([
            0.0, 0.0, 0.0,  # Position (x, y, z)
            0.0, 0.0, 0.0,  # Velocity (vx, vy, vz)
            0.0, 0.0, 0.0,  # Orientation (roll, pitch, yaw)
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ])
        self.initial_covariance = np.ones((15, 15)) * 1e-5

        self.accelerometer_noise_density = 1e-2
        self.accelerometer_random_walk = 1e-4
        self.gyroscope_noise_density = 1e-4
        self.gyroscope_random_walk = 1e-6

        self.ekf_wrapper = EKFWrapper(
            self.initial_state,
            self.initial_covariance,
            self.accelerometer_noise_density,
            self.gyroscope_noise_density,
            self.accelerometer_random_walk,
            self.gyroscope_random_walk)

        self.assertIsNotNone(
            self.ekf_wrapper, "EKFWrapper should be initialized.")
        self.assertEqual(self.ekf_wrapper.state.shape,
                         (15,), "State shape should be (15,).")
        self.assertEqual(self.ekf_wrapper.state,
                         self.initial_state, "Initial state should match.")
        self.assertEqual(self.ekf_wrapper.state_covariance.shape,
                         (15, 15), "State covariance shape should be (15, 15).")
        self.assertEqual(self.ekf_wrapper.state_covariance,
                         self.initial_covariance, "Initial covariance should match.")
        self.assertEqual(self.ekf_wrapper.imu_noise.shape,
                         (6,), "IMU noise shape should be (6,).")
        self.assertEqual(self.ekf_wrapper.imu_noise,
                         np.array([self.accelerometer_noise_density, self.accelerometer_noise_density, self.accelerometer_noise_density,
                                   self.gyroscope_noise_density, self.gyroscope_noise_density, self.gyroscope_noise_density]),
                         "IMU noise should match.")
        self.assertEqual(self.ekf_wrapper.process_noise_covariance.shape,
                         (6,), "Process noise covariance shape should be (6,).")
        self.assertEqual(self.ekf_wrapper.process_noise_covariance,
                         np.array([self.accelerometer_noise_density ** 2, self.accelerometer_noise_density ** 2, self.accelerometer_noise_density ** 2,
                                   self.gyroscope_noise_density ** 2, self.gyroscope_noise_density ** 2, self.gyroscope_noise_density ** 2]),
                         "Process noise covariance should match.")


if __name__ == '__main__':
    unittest.main()
