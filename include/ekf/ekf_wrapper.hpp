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
* @file ekf_wrapper.hpp
*
* An EKF Wrapper implementation
*
* @authors Rodrigo Da Silva Gómez
*/

#ifndef EKF__EKF_WRAPPER_HPP
#define EKF__EKF_WRAPPER_HPP

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <cmath>

#include <ekf/ekf_c_code.h>

#include "Eigen/src/Core/Matrix.h"
#include "ekf/ekf_datatype.hpp"

namespace ekf
{

/**
 * @brief EKFData
 *
 * Data structure to hold the EKF data.
 */
struct EKFData
{
  State state; // Current state of the EKF
  Covariance covariance; // Current covariance of the EKF
  Gravity gravity; // Gravity vector
  Eigen::Matrix4d map_to_odom; // Transformation matrix from map to odometry frame
};

/**
 * @brief EKFWrapper
 *
 * Class to wrap the EKF functionality.
 */
class EKFWrapper
{
public:
  EKFWrapper();
  EKFWrapper(
    State initial_state,
    Covariance initial_covariance,
    Eigen::Vector<double, 6> imu_noise,
    double accelerometer_noise_density,
    double gyroscope_noise_density,
    double accelerometer_random_walk,
    double gyroscope_random_walk);
  ~EKFWrapper();

  /**
   * @brief Reset the EKF with a new state and covariance.
   *
   * @param initial_state (State) The new initial state vector.
   * @param initial_covariance (Covariance) The new initial covariance.
   */
  void reset(
    const State & initial_state,
    const Covariance & initial_covariance);

  /**
   * @brief Set the IMU noise parameters.
   * @param imu_noise (Eigen::Vector<double, 6>) The IMU noise vector.
   * @param accelerometer_noise_density (double) The accelerometer noise density.
   * @param gyroscope_noise_density (double) The gyroscope noise density.
   * @param accelerometer_random_walk (double) The accelerometer random walk.
   * @param gyroscope_random_walk (double) The gyroscope random walk.
   * */
  void set_noise_parameters(
    const Eigen::Vector<double, 6> & imu_noise,
    double accelerometer_noise_density,
    double gyroscope_noise_density,
    double accelerometer_random_walk,
    double gyroscope_random_walk);

  /**
   * @brief Set gravity vector.
   * @param gravity (Gravity) The gravity vector.
   */
  void set_gravity(const Gravity & gravity);

  /**
   * @brief Get the current state.
   *
   * @return The current state vector.
   */
  State get_state();

  /**
   * @brief Get the current state covariance.
   *
   * @return The current state covariance matrix.
   */
  Covariance get_state_covariance();

  /**
   * @brief Get the current map to odom transformation.
   *
   * @return The current map to odom transformation matrix.
   */
  Eigen::Matrix4d get_map_to_odom();

  /**
   * @brief Get the gravity vector.
   *
   * @return The gravity vector.
   */
  Gravity get_gravity();

  /**
   * @brief Get the IMU noise vector.
   *
   * @return The IMU noise vector.
   */
  Eigen::Vector<double, 6> get_imu_noise();

  /**
   * @brief Get the noise parameters.
   *
   * @return The noise parameters as a vector.
   */
  Eigen::Vector<double, 4> get_noise_parameters();

  /**
   * @brief Compute the process noise covariance matrix.
   *
   * @param dt (double) The time step.
   * @return The process noise covariance matrix.
   */
  Covariance compute_process_noise_covariance(double dt);

  /**
   * @brief Predict the next state.
   *
   * @param imu_measurement (Input) The IMU measurement vector.
   * @param dt (double) The time step.
   */
  void predict(
    Input imu_measurement,
    double dt);

  /**
   * @brief Update the state with a new pose measurement.
   *
   * @param z (PoseMeasurement) The measurement (pose) vector.
   * @param measurement_noise_covariance (PoseMeasurementCovariance) The measurement noise covariance matrix.
   */
  void update_pose(
    const PoseMeasurement & z,
    const PoseMeasurementCovariance & measurement_noise_covariance);

  /**
   * @brief Update the state with a new pose and velocity measurement.
   *
   * @param z (PoseVelocityMeasurement) The measurement (pose and velocity) vector.
   * @param measurement_noise_covariance (PoseVelocityMeasurementCovariance) The measurement noise covariance matrix.
   */
  void update_pose_velocity(
    const PoseVelocityMeasurement & z,
    const PoseVelocityMeasurementCovariance & measurement_noise_covariance);

private:
  EKFData ekf_data_; // EKF data structure
  Eigen::Vector<double, 6> imu_noise_; // IMU noise vector
  double accelerometer_noise_density_; // Accelerometer noise density
  double gyroscope_noise_density_; // Gyroscope noise density
  double accelerometer_random_walk_; // Accelerometer random walk
  double gyroscope_random_walk_; // Gyroscope random walk

  const double * arg_[predict_function_SZ_ARG]; // Arguments for the predict functionality
  double * res_[predict_function_SZ_RES]; // Results for the predict functionality
};

} // namespace ekf

#endif // EKF__EKF_WRAPPER_HPP
