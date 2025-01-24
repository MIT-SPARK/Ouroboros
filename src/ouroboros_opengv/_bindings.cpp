#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/sac_problems/point_cloud/PointCloudSacProblem.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

namespace py = pybind11;
using namespace py::literals;

using opengv::absolute_pose::AbsoluteAdapterBase;
using opengv::point_cloud::PointCloudAdapterBase;
using opengv::relative_pose::RelativeAdapterBase;
using opengv::sac::Ransac;
using opengv::sac_problems::point_cloud::PointCloudSacProblem;

using RelativePoseProblem =
    opengv::sac_problems::relative_pose::CentralRelativePoseSacProblem;
using AbsolutePoseProblem = opengv::sac_problems::absolute_pose::AbsolutePoseSacProblem;

using Vec3Matrix = Eigen::Matrix<double, 3, Eigen::Dynamic>;

inline void checkInputs(const Vec3Matrix& m1,
                        const Vec3Matrix& m2,
                        const std::string& m1_name,
                        const std::string& m2_name) {
  if (m1.cols() == m2.cols()) {
    return;
  }

  std::stringstream ss;
  ss << "Invalid input shapes! Given " << m1_name << ": [" << m1.rows() << ", "
     << m1.cols() << "] and " << m2_name << ": [" << m2.rows() << ", " << m2.cols()
     << "] (columns do not match: " << m1.cols() << " != " << m2.cols() << ")";
  throw std::invalid_argument(ss.str());
}

struct EigenRelativeAdaptor : public RelativeAdapterBase {
  EigenRelativeAdaptor(const Vec3Matrix& src, const Vec3Matrix& dest)
      : EigenRelativeAdaptor(src, dest, Eigen::Matrix3d::Identity()) {}

  EigenRelativeAdaptor(const Vec3Matrix& src,
                       const Vec3Matrix& dest,
                       const Eigen::Matrix3d& rotation_prior)
      : EigenRelativeAdaptor(src, dest, rotation_prior, Eigen::Vector3d::Zero()) {}

  EigenRelativeAdaptor(const Vec3Matrix& _src,
                       const Vec3Matrix& _dest,
                       const Eigen::Matrix3d& rotation_prior,
                       const Eigen::Vector3d& translation_prior)
      : RelativeAdapterBase(translation_prior, rotation_prior), src(_src), dest(_dest) {
    checkInputs(src, dest, "src_bearings", "dest_bearings");
  }

  Eigen::Vector3d getBearingVector1(size_t index) const override {
    return dest.block<3, 1>(0, index);
  }

  Eigen::Vector3d getBearingVector2(size_t index) const override {
    return src.block<3, 1>(0, index);
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

  size_t getNumberCorrespondences() const override { return dest.cols(); }

  Vec3Matrix src;
  Vec3Matrix dest;
};

struct EigenAbsoluteAdaptor : public AbsoluteAdapterBase {
  EigenAbsoluteAdaptor(const Vec3Matrix& bearings, const Vec3Matrix& points)
      : EigenAbsoluteAdaptor(bearings, points, Eigen::Matrix3d::Identity()) {}

  EigenAbsoluteAdaptor(const Vec3Matrix& bearings,
                       const Vec3Matrix& points,
                       const Eigen::Matrix3d& rotation_prior)
      : EigenAbsoluteAdaptor(
            bearings, points, rotation_prior, Eigen::Vector3d::Zero()) {}

  EigenAbsoluteAdaptor(const Vec3Matrix& _bearings,
                       const Vec3Matrix& _points,
                       const Eigen::Matrix3d& rotation_prior,
                       const Eigen::Vector3d& translation_prior)
      : AbsoluteAdapterBase(translation_prior, rotation_prior),
        bearings(_bearings),
        points(_points) {
    checkInputs(bearings, points, "bearings", "points");
  }

  Eigen::Vector3d getBearingVector(size_t index) const override {
    return bearings.block<3, 1>(0, index);
  }

  // TODO(nathan) think about weighted correspondences
  double getWeight(size_t index) const override { return 1.0; }

  Eigen::Vector3d getCamOffset(size_t index) const override {
    return Eigen::Vector3d::Zero();
  }

  Eigen::Matrix3d getCamRotation(size_t index) const override {
    return Eigen::Matrix3d::Identity();
  }

  Eigen::Vector3d getPoint(size_t index) const override {
    return points.block<3, 1>(0, index);
  }

  size_t getNumberCorrespondences() const override { return bearings.cols(); }

  Vec3Matrix bearings;
  Vec3Matrix points;
};

struct EigenPointCloudAdaptor : public PointCloudAdapterBase {
  EigenPointCloudAdaptor(const Vec3Matrix& src, const Vec3Matrix& dest)
      : EigenPointCloudAdaptor(src, dest, Eigen::Matrix3d::Identity()) {}

  EigenPointCloudAdaptor(const Vec3Matrix& src,
                         const Vec3Matrix& dest,
                         const Eigen::Matrix3d& rotation_prior)
      : EigenPointCloudAdaptor(src, dest, rotation_prior, Eigen::Vector3d::Zero()) {}

  EigenPointCloudAdaptor(const Vec3Matrix& _src,
                         const Vec3Matrix& _dest,
                         const Eigen::Matrix3d& rotation_prior,
                         const Eigen::Vector3d& translation_prior)
      : PointCloudAdapterBase(translation_prior, rotation_prior),
        src(_src),
        dest(_dest) {
    checkInputs(src, dest, "src_points", "dest_points");
  }

  Eigen::Vector3d getPoint1(size_t index) const override {
    return dest.block<3, 1>(0, index);
  }

  Eigen::Vector3d getPoint2(size_t index) const override {
    return src.block<3, 1>(0, index);
  }

  size_t getNumberCorrespondences() const override { return dest.cols(); }

  // TODO(nathan) think about weighted correspondences
  double getWeight(size_t index) const override { return 1.0; }

  Vec3Matrix src;
  Vec3Matrix dest;
};

struct RansacResult {
  bool valid = false;
  Eigen::Matrix4d dest_T_src;
  std::vector<size_t> inliers;
  operator bool() const { return valid; }

  RansacResult() = default;
  RansacResult(const Ransac<RelativePoseProblem>& ransac)
      : valid(true),
        dest_T_src(Eigen::Matrix4d::Identity()),
        inliers(ransac.inliers_.begin(), ransac.inliers_.end()) {
    dest_T_src.block<3, 4>(0, 0) = ransac.model_coefficients_;
  }

  RansacResult(const Ransac<AbsolutePoseProblem>& ransac)
      : valid(true),
        dest_T_src(Eigen::Matrix4d::Identity()),
        inliers(ransac.inliers_.begin(), ransac.inliers_.end()) {
    dest_T_src.block<3, 4>(0, 0) = ransac.model_coefficients_;
  }

  RansacResult(const Ransac<PointCloudSacProblem>& ransac)
      : valid(true),
        dest_T_src(Eigen::Matrix4d::Identity()),
        inliers(ransac.inliers_.begin(), ransac.inliers_.end()) {
    dest_T_src.block<3, 4>(0, 0) = ransac.model_coefficients_;
  }
};

PYBIND11_MODULE(_ouroboros_opengv, module) {
  py::options options;
  options.disable_function_signatures();

  module.doc() = R"(Module wrapping opengv two-view geometry solvers.)";

  py::class_<RansacResult>(module, "RansacResult", "Result struct for ransac result.")
      .def(
          "__bool__",
          [](const RansacResult& result) -> bool { return result; },
          "Return whether or not the pose is valid.")
      .def_readonly(
          "valid", &RansacResult::valid, "Whether or not the pose result is valid.")
      .def_readonly("dest_T_src",
                    &RansacResult::dest_T_src,
                    "Transform from the src input to the destination input.")
      .def_readonly("inliers",
                    &RansacResult::inliers,
                    "Indicies of the inliers matching the model.");

  py::enum_<RelativePoseProblem::Algorithm>(module, "Solver2d2d")
      .value("STEWENIUS", RelativePoseProblem::Algorithm::STEWENIUS)
      .value("NISTER", RelativePoseProblem::Algorithm::NISTER)
      .value("SEVENPT", RelativePoseProblem::Algorithm::SEVENPT)
      .value("EIGHTPT", RelativePoseProblem::Algorithm::EIGHTPT)
      .export_values();

  // NOTE(nathan) we don't expose EPNP because it spams stdout
  py::enum_<AbsolutePoseProblem::Algorithm>(module, "Solver2d3d")
      .value("TWOPT", AbsolutePoseProblem::TWOPT)
      .value("KNEIP", AbsolutePoseProblem::Algorithm::KNEIP)
      .value("GAO", AbsolutePoseProblem::Algorithm::GAO)
      .value("GP3P", AbsolutePoseProblem::Algorithm::GP3P)
      .export_values();

  // TODO(nathan) allow for prior rotation or translation
  module.def(
      "solve_2d2d",
      [](const Vec3Matrix& src,
         const Vec3Matrix& dest,
         RelativePoseProblem::Algorithm solver,
         size_t max_iterations,
         double threshold,
         double probability) -> RansacResult {
        EigenRelativeAdaptor adaptor(src, dest);
        Ransac<RelativePoseProblem> ransac;
        ransac.max_iterations_ = max_iterations;
        ransac.threshold_ = threshold;
        ransac.probability_ = probability;
        ransac.sac_model_ = std::make_shared<RelativePoseProblem>(adaptor, solver);
        if (!ransac.computeModel()) {
          return {};
        }

        return ransac;
      },
      R"(Recover two-view pose up-to-scale via feature correspondences.

Args:
  src (numpy.ndarray[numpy.float64[3, n]]): Feature bearings arranged as (3, N) for the "src" input.
  dest (numpy.ndarray[numpy.float64[3, n]]): Feature bearings arranged as (3, N) for the "dest" input.
  solver (_ouroboros_opengv.Solver2d2d): Minimal solver type to use.
  max_iterations (int): Maximum RANSAC iterations.
  threshold (float): Inlier error threshold.
  probability (float): Likelihood that minimal indices contain at least one inlier.

Returns:
  _ouroboros_opengv.RansacResult: Potentially valid dest_T_src and associated inliers)",
      "src"_a,
      "dest"_a,
      "solver"_a = RelativePoseProblem::Algorithm::STEWENIUS,
      "max_iterations"_a = 1000,
      "threshold"_a = 1.0e-2,
      "probability"_a = 0.99);

  module.def(
      "solve_2d3d",
      [](const Vec3Matrix& bearings,
         const Vec3Matrix& points,
         AbsolutePoseProblem::Algorithm solver,
         size_t max_iterations,
         double threshold,
         double probability) -> RansacResult {
        EigenAbsoluteAdaptor adaptor(bearings, points);
        Ransac<AbsolutePoseProblem> ransac;
        ransac.max_iterations_ = max_iterations;
        ransac.threshold_ = threshold;
        ransac.probability_ = probability;
        ransac.sac_model_ = std::make_shared<AbsolutePoseProblem>(adaptor, solver);

        if (!ransac.computeModel()) {
          return {};
        }

        return ransac;
      },
      R"(Recover two-view pose via PnP.

Args:
  bearings (numpy.ndarray[numpy.float64[3, n]]): Feature bearings arranged as (3, N) for the "src" input.
  points (numpy.ndarray[numpy.float64[3, n]]): Corresponding points arranged as (3, N) for the "dest" input.
  solver (_ouroboros_opengv.Solver2d3d): PnP minimal solver type to use.
  max_iterations (int): Maximum RANSAC iterations.
  threshold (float): Inlier error threshold.
  probability (float): Likelihood that minimal indices contain at least one inlier.

Returns:
  _ouroboros_opengv.RansacResult: Potentially valid dest_T_src and associated inliers)",
      "bearings"_a,
      "points"_a,
      "solver"_a = AbsolutePoseProblem::Algorithm::KNEIP,
      "max_iterations"_a = 1000,
      "threshold"_a = 1.0e-2,
      "probability"_a = 0.99);

  module.def(
      "solve_3d3d",
      [](const Vec3Matrix& src,
         const Vec3Matrix& dest,
         size_t max_iterations,
         double threshold,
         double probability) -> RansacResult {
        EigenPointCloudAdaptor adaptor(src, dest);
        Ransac<PointCloudSacProblem> ransac;
        ransac.max_iterations_ = max_iterations;
        ransac.threshold_ = threshold;
        ransac.probability_ = probability;
        ransac.sac_model_ = std::make_shared<PointCloudSacProblem>(adaptor);
        if (!ransac.computeModel()) {
          return {};
        }

        return ransac;
      },
      R"(Recover two-view pose via Arun's method.

Args:
  src (numpy.ndarray[numpy.float64[3, n]]): Point cloud for the "src" input.
  dest (numpy.ndarray[numpy.float64[3, n]]): Point cloud arranged as (3, N) for the "dest" input.
  max_iterations (int): Maximum RANSAC iterations.
  threshold (float): Inlier error threshold.
  probability (float): Likelihood that minimal indices contain at least one inlier.

Returns:
  _ouroboros_opengv.RansacResult: Potentially valid dest_T_src and associated inliers)",

      "src"_a,
      "dest"_a,
      "max_iterations"_a = 1000,
      "threshold"_a = 1.0e-2,
      "probability"_a = 0.99);
}
