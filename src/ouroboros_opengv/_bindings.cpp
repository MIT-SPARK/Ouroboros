#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  EigenRelativeAdaptor(const Eigen::MatrixXd& bearings1,
                       const Eigen::MatrixXd& bearings2)
      : EigenRelativeAdaptor(bearings1, bearings2, Eigen::Matrix3d::Identity()) {}

  EigenRelativeAdaptor(const Eigen::MatrixXd& bearings1,
                       const Eigen::MatrixXd& bearings2,
                       const Eigen::Matrix3d& rotation_prior)
      : EigenRelativeAdaptor(
            bearings1, bearings2, rotation_prior, Eigen::Vector3d::Zero()) {}

  EigenRelativeAdaptor(const Eigen::MatrixXd& _bearings1,
                       const Eigen::MatrixXd& _bearings2,
                       const Eigen::Matrix3d& rotation_prior,
                       const Eigen::Vector3d& translation_prior)
      : RelativeAdapterBase(translation_prior, rotation_prior),
        bearings1(_bearings1),
        bearings2(_bearings2) {
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

  // TODO(nathan) think about weighted correspondences
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

struct RansacResult {
  bool valid = false;
  Eigen::Matrix4d dest_T_src;
  std::vector<size_t> inliers;
  operator bool() const { return valid; }

  RansacResult() = default;
  RansacResult(const Ransac<CentralRelativePoseSacProblem>& ransac)
      : valid(true),
        dest_T_src(Eigen::Matrix4d::Identity()),
        inliers(ransac.inliers_.begin(), ransac.inliers_.end()) {
    dest_T_src.block<3, 4>(0, 0) = ransac.model_coefficients_;
  }
};

PYBIND11_MODULE(_ouroboros_opengv, module) {
  py::enum_<CentralRelativePoseSacProblem::Algorithm>(module, "RelativeSolver")
      .value("STEWENIUS", CentralRelativePoseSacProblem::Algorithm::STEWENIUS)
      .value("NISTER", CentralRelativePoseSacProblem::Algorithm::NISTER)
      .value("SEVENPT", CentralRelativePoseSacProblem::Algorithm::SEVENPT)
      .value("EIGHTPT", CentralRelativePoseSacProblem::Algorithm::EIGHTPT)
      .export_values();

  py::class_<RansacResult>(module, "RansacResult")
      .def(py::init<>())
      .def("__bool__", [](const RansacResult& result) -> bool { return result; })
      .def_readonly("valid", &RansacResult::valid)
      .def_readonly("dest_T_src", &RansacResult::dest_T_src)
      .def_readonly("inliers", &RansacResult::inliers);

  // TODO(nathan) allow for prior rotation or translation
  module.def(
      "solve_2d2d",
      [](const Eigen::MatrixXd& dest,
         const Eigen::MatrixXd& src,
         CentralRelativePoseSacProblem::Algorithm solver,
         size_t max_iterations,
         double threshold,
         double probability) -> RansacResult {
        EigenRelativeAdaptor adaptor(dest, src);
        Ransac<CentralRelativePoseSacProblem> ransac;
        ransac.max_iterations_ = max_iterations;
        ransac.threshold_ = threshold;
        ransac.probability_ = probability;
        ransac.sac_model_ =
            std::make_shared<CentralRelativePoseSacProblem>(adaptor, solver);

        if (!ransac.computeModel()) {
          return {};
        }

        return ransac;
      },
      "dest"_a,
      "src"_a,
      "solver"_a = CentralRelativePoseSacProblem::Algorithm::STEWENIUS,
      "max_iterations"_a = 1000,
      "threshold"_a = 1.0e-2,
      "probability"_a = 0.99);
}
