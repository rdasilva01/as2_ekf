// Copyright 2025 Universidad Politécnica de Madrid
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//    * Redistributions of source code must retain the above copyright
//      notice, this list of conditions and the following disclaimer.
//
//    * Redistributions in binary form must reproduce the above copyright
//      notice, this list of conditions and the following disclaimer in the
//      documentation and/or other materials provided with the distribution.
//
//    * Neither the name of the Universidad Politécnica de Madrid nor the names of its
//      contributors may be used to endorse or promote products derived from
//      this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

/**
* @file ekf_datatype.hpp
*
* An EKF Wrapper implementation
*
* @authors Rodrigo Da Silva Gómez
*/

#ifndef EKF__EKF_DATATYPE_H
#define EKF__EKF_DATATYPE_H

#include <array>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace ekf
{

/**
 * @brief State X
 */
struct State
{
  static const std::size_t size = 15;
  std::array<double, size> data;

  /**
   * @brief Constructor
   */
  State();

  /**
   * @brief Constructor with initial values
   * @param values Initial values for the state
   */
  State(const std::array<double, size> & values);

  /**
   * @brief Sets the state to the provided values
   * @param values Values to set the state
   */
  void set(const std::array<double, size> & values);

  /**
   * @brief Get position (x, y, z)
   * @return A 3D vector representing the position
   */
  std::array<double, 3> get_position() const;

  /**
   * @brief Get velocity (vx, vy, vz)
   * @return A 3D vector representing the Velocity
   */
  std::array<double, 3> get_velocity() const;

  /**
   * @brief Get orientation (roll, pitch, yaw)
   * @return A 3D vector representing the orientation in radians
   */
  std::array<double, 3> get_orientation() const;

  /**
   * @brief Get accelerometer bias (abx, aby, abz)
   * @return A 3D vector representing the accelerometer bias
   */
  std::array<double, 3> get_accelerometer_bias() const;

  /**
   * @brief Get gyroscope bias (wbx, wby, wbz)
   * @return A 3D vector representing the gyroscope bias
   */
  std::array<double, 3> get_gyroscope_bias() const;

  /**
   * @brief The print operator for easy debugging
   * @return A string representation of the state
   */
  std::string to_string() const;

};


/**
 * @brief State covariance P
 */
struct Covariance
{
  static const std::size_t size = 225; // 15x15 covariance matrix
  std::array<double, size> data;

  /**
   * @brief Constructor
   */
  Covariance();

  /**
   * @brief Constructor with initial values
   * @param values Initial values for the covariance
   */
  Covariance(const std::array<double, size> & values);

  /**
   * @brief Sets the covariance to the provided values
   * @param values Values to set the covariance
   */
  void set(const std::array<double, size> & values);

  /**
   * @brief The print operator for easy debugging
   * @return A string representation of the state
   */
  std::string to_string() const;
};


/**
 * @brief gravity vector
 */
struct Gravity
{
  static const std::size_t size = 3;
  std::array<double, size> data;

  /**
   * @brief Constructor
   */
  Gravity();

  /**
   * @brief Constructor with initial values
   * @param values Initial values for the gravity vector
   */
  Gravity(const std::array<double, size> & values);

  /**
   * @brief Sets the gravity vector to the provided values
   * @param values Values to set the gravity vector
   */
  void set(const std::array<double, size> & values);
};


/**
 * @brief IMU input measurements
 */
struct Input
{
  static const std::size_t size = 6; // 3 accelerometer + 3 gyroscope
  std::array<double, size> data;

  /**
   * @brief Constructor
   */
  Input();

  /**
   * @brief Constructor with initial values
   * @param values Initial values for the input measurements
   */
  Input(const std::array<double, size> & values);

  /**
   * @brief Sets the input measurements to the provided values
   * @param values Values to set the input measurements
   */
  void set(const std::array<double, size> & values);
};


/**
 * @brief Pose measurement Z_pose
 */
struct PoseMeasurement
{
  static const std::size_t size = 6; // 3 position + 3 orientation (quaternion)
  std::array<double, size> data;

  /**
   * @brief Constructor
   */
  PoseMeasurement();

  /**
   * @brief Constructor with initial values
   * @param values Initial values for the pose measurement
   */
  PoseMeasurement(const std::array<double, size> & values);

  /**
   * @brief Sets the pose measurement to the provided values
   * @param values Values to set the pose measurement
   */
  void set(const std::array<double, size> & values);
};


/**
 * @brief Pose measurement covariance diagonal R_pose
 */
struct PoseMeasurementCovariance
{
  static const std::size_t size = 6; // 3 position + 3 orientation (quaternion)
  std::array<double, size> data;

  /**
   * @brief Constructor
   */
  PoseMeasurementCovariance();

  /**
   * @brief Constructor with initial values
   * @param values Initial values for the pose measurement covariance
   */
  PoseMeasurementCovariance(const std::array<double, size> & values);

  /**
   * @brief Sets the pose measurement covariance to the provided values
   * @param values Values to set the pose measurement covariance
   */
  void set(const std::array<double, size> & values);
};


/**
 * @brief Pose and Velocity measurement Z_pose_velocity
 */
struct PoseVelocityMeasurement
{
  static const std::size_t size = 9; // 3 position + 3 orientation (quaternion) + 3 velocity
  std::array<double, size> data;

  /**
   * @brief Constructor
   */
  PoseVelocityMeasurement();

  /**
   * @brief Constructor with initial values
   * @param values Initial values for the pose and velocity measurement
   */
  PoseVelocityMeasurement(const std::array<double, size> & values);

  /**
   * @brief Sets the pose and velocity measurement to the provided values
   * @param values Values to set the pose and velocity measurement
   */
  void set(const std::array<double, size> & values);
};


/**
 * @brief Pose and Velocity measurement covariance diagonal R_pose_velocity
 */
struct PoseVelocityMeasurementCovariance
{
  static const std::size_t size = 9; // 3 position + 3 orientation (quaternion) + 3 velocity
  std::array<double, size> data;

  /**
   * @brief Constructor
   */
  PoseVelocityMeasurementCovariance();

  /**
   * @brief Constructor with initial values
   * @param values Initial values for the pose and velocity measurement covariance
   */
  PoseVelocityMeasurementCovariance(const std::array<double, size> & values);

  /**
   * @brief Sets the pose and velocity measurement covariance to the provided values
   * @param values Values to set the pose and velocity measurement covariance
   */
  void set(const std::array<double, size> & values);
};

} // namespace ekf

#endif // EKF__EKF_DATATYPE_H
