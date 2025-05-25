/**
 * File: FClass.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: generic FClass to instantiate templated classes
 * License: see the LICENSE.txt file
 *
 */

#pragma once

#include <Eigen/Dense>

#include <cstdint>
#include <string>
#include <vector>

namespace DBoW3 {

class DescManip {
 public:
  static void meanValue(const Eigen::MatrixXf& descriptors, cv::Mat& mean);

  static double distance(const cv::Mat& a, const cv::Mat& b);

  static uint32_t distance_8uc1(const cv::Mat& a, const cv::Mat& b);
};

}  // namespace DBoW3
