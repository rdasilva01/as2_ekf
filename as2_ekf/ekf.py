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


import casadi as ca
from casadi_utils import Utils


class EKF():
    """
    Extended Kalman Filter (EKF) class.
    """

    def __init__(self):
        """
        Initialize the EKF.
        """

        # Time step
        self.dt = ca.SX.sym('dt')
        self.g = ca.DM(9.81)  # Gravity constant

        # State vector
        # x, y, z, vx, vy, vz, roll, pitch, yaw, abx, aby, abz, wbx, wby, wbz
        self.X = ca.SX.sym('X', 15)
        # self.x, self.y, self.z, \
        #     self.vx, self.vy, self.vz, \
        #     self.qw, self.qx, self.qy, self.qz, \
        #     self.abx, self.aby, self.abz, \
        #     self.wbx, self.wby, self.wbz = ca.vertsplit(self.X)
        state_position = self.X[0:3]
        state_velocity = self.X[3:6]
        state_orientation = self.X[6:9]
        state_accelerometer_bias = self.X[9:12]
        state_gyrometer_bias = self.X[12:15]

        # Inputs
        # axm, aym, azm, wxm, wym, wzm
        self.U = ca.SX.sym('U', 6)
        # self.axm, self.aym, self.azm, \
        #     self.wxm, self.wym, self.wzm = ca.vertsplit(self.U)
        input_acceleration = self.U[0:3]
        input_angular_velocity = self.U[3:6]

        # Inputs noise
        # axw, ayw, azw, wxw, wyw, wzw
        self.W = ca.SX.sym('W', 6)
        # self.axw, self.ayw, self.azw, \
        #     self.wxw, self.wyw, self.wzw = ca.vertsplit(self.W)
        input_noise_acceleration = self.W[0:3]
        input_noise_angular_velocity = self.W[3:6]

        # Inputs without noise
        # iax, iay, iaz, iwx, iwy, iwz
        self.IN = self.U - self.W
        # self.iax, self.iay, self.iaz, \
        #     self.iwx, self.iwy, self.iwz = ca.vertsplit(self.IN)
        input_wo_noise_acceleration = self.IN[0:3]
        input_wo_noise_angular_velocity = self.IN[3:6]

        # Derivatives
        p_dot = state_velocity

        v_dot = Utils.velocity_derivative(
            state_orientation,
            input_wo_noise_acceleration,
            self.g)

        # q_dot = Utils.quaternion_derivate(
        #     state_orientation,
        #     input_wo_noise_angular_velocity
        # )
        q_dot = [0, 0, 0]

        self.f_continuous = ca.vertcat(
            p_dot,
            v_dot,
            q_dot,
            0, 0, 0,
            0, 0, 0
        )

        # Discrete state transition function
        self.f = self.X + self.f_continuous * self.dt

        # Output function
        # x, y, z, roll, pitch, yaw
        self.h = ca.vertcat(
            state_position,
            # Utils.normalize_quaternion(state_orientation)
            state_orientation,
        )

        # Jacobians
        self.F = ca.jacobian(self.f, self.X)
        self.L = ca.jacobian(self.f, self.W)
        self.H = ca.jacobian(self.h, self.X)

        # Substitute W with 0
        self.f = ca.substitute(self.f, self.W, 0)
        self.F = ca.substitute(self.F, self.W, 0)
        self.L = ca.substitute(self.L, self.W, 0)
        self.H = ca.substitute(self.H, self.W, 0)

        # covariance and extra matrices
        # State covariance matrix
        self.P = ca.SX.sym('P', self.X.size()[0], self.X.size()[0])

        # Process Noise covariance matrix
        self.aux_Q_vector = ca.SX.sym('Q', self.W.size()[0])
        self.Q = ca.SX.zeros(self.W.size()[0], self.W.size()[0])
        for i in range(self.W.size()[0]):
            for j in range(i, self.W.size()[0]):
                if i == j:
                    self.Q[i, j] = self.aux_Q_vector[i]

        # Measurement Noise covariance matrix
        self.aux_R_vector = ca.SX.sym('R', self.h.size()[0])
        self.R = ca.SX.zeros(self.h.size()[0], self.h.size()[0])
        for i in range(self.h.size()[0]):
            for j in range(i, self.h.size()[0]):
                if i == j:
                    self.R[i, j] = self.aux_R_vector[i]

        self.Z = ca.SX.sym('Z', self.h.size()[0])

        # Predict step
        self.X_pred = self.f
        self.P_pred = self.F @ self.P @ self.F.T + self.L @ self.Q @ self.L.T

        # Update step
        self.Y_residual = self.Z - self.h
        self.S = self.H @ self.P @ self.H.T + self.R
        self.K = self.P @ self.H.T @ ca.pinv(self.S)
        # self.X_update = Utils.state_quaternion_normalization(
        #     self.X + self.K @ self.Y_residual)
        self.X_update = self.X + self.K @ self.Y_residual
        self.P_update = (
            ca.SX.eye(self.X.size()[0]) - self.K @ self.H) @ self.P

        # Functions
        # Define the CasADi function for prediction
        self.predict_function = ca.Function(
            'predict_function',
            [self.X, self.U, self.W, self.dt, self.P, self.aux_Q_vector],
            [self.X_pred, self.P_pred],
            ['X', 'U', 'W', 'dt', 'P', 'Q'],
            ['X_pred', 'P_pred']
        )
        # Define the CasADi function for update
        self.update_function = ca.Function(
            'update_function',
            [self.X, self.W, self.Z, self.P, self.aux_R_vector],
            [self.X_update, self.P_update],
            ['X', 'W', 'Z', 'P', 'R'],
            ['X_update', 'P_update']
        )
