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

#include "ekf/ekf_wrapper.hpp"
#include "ekf_datatype.hpp"
#include <algorithm>

namespace ekf
{

EKFWrapper::EKFWrapper()
{
  // Initialize the EKF data
  ekf_data_ = EKFData();
  ekf_data_.map_to_odom = Eigen::Matrix4d::Identity();
  imu_noise_ = Eigen::Vector<double, 6>::Zero();
  accelerometer_noise_density_ = 0.0;
  gyroscope_noise_density_ = 0.0;
  accelerometer_random_walk_ = 0.0;
  gyroscope_random_walk_ = 0.0;
  arg_[0] = ekf_data_.state.data.data();
  arg_[2] = imu_noise_.data();
  arg_[4] = ekf_data_.covariance.data.data();
  arg_[6] = ekf_data_.gravity.data.data();
  res_[0] = ekf_data_.state.data.data();
  res_[1] = ekf_data_.covariance.data.data();
  Gravity vector3;
  res_[2] = vector3.data.data();
}

EKFWrapper::EKFWrapper(
  State initial_state,
  Covariance initial_covariance,
  Eigen::Vector<double, 6> imu_noise,
  double accelerometer_noise_density,
  double gyroscope_noise_density,
  double accelerometer_random_walk,
  double gyroscope_random_walk)
{
  // Initialize the EKF data with provided parameters
  ekf_data_ = EKFData();
  ekf_data_.state = initial_state;
  ekf_data_.covariance = initial_covariance;
  ekf_data_.map_to_odom = Eigen::Matrix4d::Identity();
  imu_noise_ = imu_noise;
  accelerometer_noise_density_ = accelerometer_noise_density;
  gyroscope_noise_density_ = gyroscope_noise_density;
  accelerometer_random_walk_ = accelerometer_random_walk;
  gyroscope_random_walk_ = gyroscope_random_walk;
  arg_[0] = ekf_data_.state.data.data();
  arg_[2] = imu_noise_.data();
  arg_[4] = ekf_data_.covariance.data.data();
  arg_[6] = ekf_data_.gravity.data.data();
  res_[0] = ekf_data_.state.data.data();
  res_[1] = ekf_data_.covariance.data.data();
  Gravity vector3;
  res_[2] = vector3.data.data();
}

EKFWrapper::~EKFWrapper()
{
  // Destructor logic if needed
}

void EKFWrapper::reset(
  const State & initial_state,
  const Covariance & initial_covariance)
{
  ekf_data_.state = initial_state;
  ekf_data_.covariance = initial_covariance;
  ekf_data_.map_to_odom = Eigen::Matrix4d::Identity();
}

void EKFWrapper::set_noise_parameters(
  const Eigen::Vector<double, 6> & imu_noise,
  double accelerometer_noise_density,
  double gyroscope_noise_density,
  double accelerometer_random_walk,
  double gyroscope_random_walk)
{
  imu_noise_ = imu_noise;
  accelerometer_noise_density_ = accelerometer_noise_density;
  gyroscope_noise_density_ = gyroscope_noise_density;
  accelerometer_random_walk_ = accelerometer_random_walk;
  gyroscope_random_walk_ = gyroscope_random_walk;

  arg_[2] = imu_noise_.data();
}

void EKFWrapper::set_gravity(const Gravity & gravity)
{
  ekf_data_.gravity = gravity;
  arg_[6] = ekf_data_.gravity.data.data();
}

State EKFWrapper::get_state()
{
  return ekf_data_.state;
}

Covariance EKFWrapper::get_state_covariance()
{
  return ekf_data_.covariance;
}

Eigen::Matrix4d EKFWrapper::get_map_to_odom()
{
  return ekf_data_.map_to_odom;
}

Gravity EKFWrapper::get_gravity()
{
  return ekf_data_.gravity;
}

Eigen::Vector<double, 6> EKFWrapper::get_imu_noise()
{
  return imu_noise_;
}

Eigen::Vector<double, 4> EKFWrapper::get_noise_parameters()
{
  return Eigen::Vector<double, 4>(
    accelerometer_noise_density_,
    gyroscope_noise_density_,
    accelerometer_random_walk_,
    gyroscope_random_walk_);
}

Covariance EKFWrapper::compute_process_noise_covariance(
  double dt)
{
  Eigen::Matrix<double, 15, 15> process_noise_covariance =
    Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix3d q_pp =
    pow(accelerometer_noise_density_, 2) *
    pow(dt, 3) /
    3.0 *
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d q_pv =
    pow(accelerometer_noise_density_, 2) *
    pow(dt, 2) /
    2.0 *
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d q_vv =
    pow(accelerometer_noise_density_, 2) *
    dt *
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d q_ww =
    pow(gyroscope_noise_density_, 2) *
    dt *
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d q_baba =
    pow(accelerometer_random_walk_, 2) *
    dt *
    Eigen::Matrix3d::Identity();
  Eigen::Matrix3d q_bwbw =
    pow(gyroscope_random_walk_, 2) *
    dt *
    Eigen::Matrix3d::Identity();
  process_noise_covariance.block<3, 3>(0, 0) = q_pp;
  process_noise_covariance.block<3, 3>(0, 3) = q_pv;
  process_noise_covariance.block<3, 3>(3, 0) = q_pv;
  process_noise_covariance.block<3, 3>(3, 3) = q_vv;
  process_noise_covariance.block<3, 3>(6, 6) = q_ww;
  process_noise_covariance.block<3, 3>(9, 9) = q_baba;
  process_noise_covariance.block<3, 3>(12, 12) = q_bwbw;
  Covariance pnc = Covariance();
  std::array<double, Covariance::size> process_noise_covariance_array;
  for (std::size_t i = 0; i < Covariance::size; ++i) {
    process_noise_covariance_array[i] = process_noise_covariance(i / 15, i % 15);
  }
  pnc.set(process_noise_covariance_array);
  return pnc;
}

void EKFWrapper::predict(
  Input input,
  double dt)
{
  Covariance process_noise_covariance =
    compute_process_noise_covariance(dt);

  arg_[1] = input.data.data();
  arg_[3] = &dt;
  arg_[5] = process_noise_covariance.data.data();

  // for (std::size_t i = 0; i < 7; ++i) {
  //   const casadi_int * arg_size_arr = predict_function_sparsity_in(i);
  //   int arg_size = arg_size_arr[0] * arg_size_arr[1];
  //   for (int j = 0; j < arg_size; ++j) {
  //     std::cout << "arg_[" << i << "][" << j << "] = " << arg_[i][j] << std::endl;
  //   }
  // }

  // // Check for null pointers
  // for (std::size_t i = 0; i < 7; ++i) {
  //   if (arg_[i] == nullptr) {
  //     std::cerr << "Error: arg_[" << i << "] is null." << std::endl;
  //     return;
  //   }
  // }

  predict_function(
    arg_,
    res_,
    0,
    0,
    0);

}


}  // namespace ekf
