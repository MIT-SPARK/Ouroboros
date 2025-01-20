#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

//#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
//#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/relative_pose/RelativeAdapterBase.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

namespace py = pybind11;
using namespace py::literals;

using opengv::relative_pose::RelativeAdapterBase;
using opengv::sac::Ransac;
using opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;

struct EigenRelativeAdaptor : public RelativeAdapterBase {
  EigenRelativeAdaptor(const Eigen::MatrixXd &bearings1,
                       const Eigen::MatrixXd &bearings2)
      : EigenRelativeAdaptor(bearings1, bearings2,
                             Eigen::Matrix3d::Identity()) {}

  EigenRelativeAdaptor(const Eigen::MatrixXd &bearings1,
                       const Eigen::MatrixXd &bearings2,
                       const Eigen::Matrix3d &rotation_prior)
      : EigenRelativeAdaptor(bearings1, bearings2, rotation_prior,
                             Eigen::Vector3d::Zero()) {}

  EigenRelativeAdaptor(const Eigen::MatrixXd &_bearings1,
                       const Eigen::MatrixXd &_bearings2,
                       const Eigen::Matrix3d &rotation_prior,
                       const Eigen::Vector3d &translation_prior)
      : RelativeAdapterBase(translation_prior, rotation_prior),
        bearings1(_bearings1), bearings2(_bearings2) {
    if (bearings1.rows() != 3) {
      throw std::invalid_argument("Invalid shape for bearings1");
    }

    if (bearings2.rows() != 3) {
      throw std::invalid_argument("Invalid shape for bearings2");
    }

    if (bearings1.cols() != bearings2.cols()) {
      throw std::invalid_argument("Number of bearings do not match");
    }
  }

  Eigen::Vector3d getBearingVector1(size_t index) const override {
    return bearings1.block<3, 1>(0, index);
  }

  Eigen::Vector3d getBearingVector2(size_t index) const override {
    return bearings2.block<3, 1>(0, index);
  }

  double getWeight(size_t index) const override { return 1.0; }

  Eigen::Vector3d getCamOffset1(size_t index) const override {
    return Eigen::Vector3d::Zero();
  }

  Eigen::Matrix3d getCamRotation1(size_t index) const override {
    return Eigen::Matrix3d::Identity();
  }

  Eigen::Vector3d getCamOffset2(size_t index) const override {
    return Eigen::Vector3d::Zero();
  }

  Eigen::Matrix3d getCamRotation2(size_t index) const override {
    return Eigen::Matrix3d::Identity();
  }

  size_t getNumberCorrespondences() const override { return bearings1.cols(); }

  Eigen::MatrixXd bearings1;
  Eigen::MatrixXd bearings2;
};

PYBIND11_MODULE(_ouroboros_opengv, module) {
  module.def("foo", [](int a, int b) { return a + 2 * b; });

  module.def(
      "solve",
      [](const Eigen::MatrixXd &a, const Eigen::MatrixXd &b, float threshold,
         size_t max_iterations) {
        EigenRelativeAdaptor adaptor(a, b);
        Ransac<CentralRelativePoseSacProblem> ransac;
        ransac.sac_model_ = std::make_shared<CentralRelativePoseSacProblem>(
            adaptor, CentralRelativePoseSacProblem::NISTER);
        ransac.threshold_ = threshold;
        ransac.max_iterations_ = max_iterations;
        ransac.computeModel();
        return ransac.model_coefficients_;
      },
      "a"_a, "b"_a, "threshold"_a = 1.0e-2, "max_iterations"_a = 1000);
}
