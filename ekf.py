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


class Utils():
    """
    Utility functions for the EKF.
    """

    @staticmethod
    def quaternion_multiply(q1: ca.SX, q2: ca.SX) -> ca.SX:
        """
        Multiply two quaternions.

        q = q1 x q2 = [qw1, qx1, qy1, qz1] x [qw2, qx2, qy2, qz2]

        :param q1 (ca.SX): The first quaternion [qw1, qx1, qy1, qz1].
        :param q2 (ca.SX): The second quaternion [qw2, qx2, qy2, qz2].

        :return (ca.SX): The resulting quaternion [qw, qx, qy, qz].
        """
        qw1 = q1[0]
        qx1 = q1[1]
        qy1 = q1[2]
        qz1 = q1[3]

        qw2 = q2[0]
        qx2 = q2[1]
        qy2 = q2[2]
        qz2 = q2[3]

        qw = qw1 * qw2 - qx1 * qx2 - qy1 * qy2 - qz1 * qz2
        qx = qw1 * qx2 + qx1 * qw2 + qy1 * qz2 - qz1 * qy2
        qy = qw1 * qy2 - qx1 * qz2 + qy1 * qw2 + qz1 * qx2
        qz = qw1 * qz2 + qx1 * qy2 - qy1 * qx2 + qz1 * qw2

        return ca.vertcat(qw, qx, qy, qz)

    @staticmethod
    def apply_rotation(q: ca.SX, v: ca.SX) -> ca.SX:
        """
        Apply a rotation to a vector.

        v_rotated = q x v x q_conj

        :param q (ca.SX): The quaternion [qw, qx, qy, qz].
        :param v (ca.SX): The vector [vx, vy, vz].

        :return (ca.SX): The rotated vector [vx_rotated, vy_rotated, vz_rotated].
        """
        qw = q[0]
        qx = q[1]
        qy = q[2]
        qz = q[3]

        vx = v[0]
        vy = v[1]
        vz = v[2]

        q_conj = ca.vertcat(qw, -qx, -qy, -qz)

        v_rotated = Utils.quaternion_multiply(
            Utils.quaternion_multiply(q, ca.vertcat(0, vx, vy, vz)),
            q_conj)

        return ca.vertcat(v_rotated[1], v_rotated[2], v_rotated[3])

    @staticmethod
    def normalize_quaternion(q: ca.SX) -> ca.SX:
        """
        Normalize a quaternion.

        :param q (ca.SX): The quaternion to normalize.

        :return (ca.SX): The normalized quaternion.
        """
        q_norm = ca.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
        return q / q_norm

    @staticmethod
    def quaternion_derivate(quaternion: ca.SX, angular_velocity: ca.SX) -> ca.SX:
        """
        Compute the quaternion derivative.

        q_dot = 0.5 * q x omega = 0.5 * [qw, qx, qy, qz] * [0, ,wx, wy, wz]

        :param quaternion (ca.SX): The quaternion [qw, qx, qy, qz].
        :param angular_velocity (ca.SX): The angular velocity [wx, wy, wz].

        :return (ca.SX): The quaternion derivative [qw_dot, qx_dot, qy_dot, qz_dot].
        """
        w_qx = angular_velocity[0]
        w_qy = angular_velocity[1]
        w_qz = angular_velocity[2]
        w_q = ca.vertcat(
            0.0,
            w_qx,
            w_qy,
            w_qz)

        return 0.5 * Utils.quaternion_multiply(
            Utils.normalize_quaternion(quaternion), w_q)

    @staticmethod
    def velocity_derivative(
            state_orientation: ca.SX,
            input_acceleration: ca.SX,
            gravity: ca.SX) -> ca.SX:
        """
        Compute the velocity derivative.
        v_dot = q(input - noise) - g
        :return (ca.SX): The velocity derivative [vx_dot, vy_dot, vz_dot].
        """
        qw, qx, qy, qz = ca.vertsplit(state_orientation)
        iax, iay, iaz = ca.vertsplit(input_acceleration)

        v_dot = Utils.apply_rotation(
            state_orientation,
            input_acceleration
        )

        return ca.vertcat(
            v_dot[0],
            v_dot[1],
            v_dot[2] - gravity
        )


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
        # x, y, z, vx, vy, vz, qw, qx, qy, qz, abx, aby, abz, wbx, wby, wbz
        self.X = ca.SX.sym('X', 16)
        # self.x, self.y, self.z, \
        #     self.vx, self.vy, self.vz, \
        #     self.qw, self.qx, self.qy, self.qz, \
        #     self.abx, self.aby, self.abz, \
        #     self.wbx, self.wby, self.wbz = ca.vertsplit(self.X)
        state_position = self.X[0:3]
        state_velocity = self.X[3:6]
        state_orientation = self.X[6:10]
        state_accelerometer_bias = self.X[10:13]
        state_gyrometer_bias = self.X[13:16]

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

        q_dot = Utils.quaternion_derivate(
            state_orientation,
            input_wo_noise_angular_velocity
        )

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
        # x, y, z, qw, qx, qy, qz
        self.h = ca.vertcat(
            state_position,
            Utils.normalize_quaternion(state_orientation)
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
        # def symmetric_indexing(i, j): return int(
        #     i*(i+1)/2+j) if i >= j else int(j*(j+1)/2+i)
        # self.aux_P_vector = ca.SX.sym('P', symmetric_indexing(
        #     self.X.size()[0], self.X.size()[0]))
        # self.P = ca.SX.zeros(self.X.size()[0], self.X.size()[0])
        # for i in range(self.X.size()[0]):
        #     for j in range(i, self.X.size()[0]):
        #         self.P[i, j] = self.aux_P_vector[symmetric_indexing(i, j)]
        #         self.P[j, i] = self.P[i, j]
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
            [self.X_update, self.P_update, self.K],
            ['X', 'W', 'Z', 'P', 'R'],
            ['X_update', 'P_update', 'Y_residual']
        )
