#pragma once

#include <algorithm>
#include <utility>

#include <Eigen/Core>

namespace shared_kinematics {

// When two segment axes count as parallel.
enum class ParallelTest {
  kAngle,       // the squared sine of the angle between the axes is below 1e-12
  kFranka32,    // a * e - b * b <= 1e-12, as positronic-franka#32 tests it; the self pairs keep it
};

// The closest points of segments [p1, q1] and [p2, q2] (Ericson, Real-Time Collision Detection, 5.1.9).
// A segment of zero length is a point.
inline std::pair<Eigen::Vector3d, Eigen::Vector3d> closest_points_of_segments(
    const Eigen::Vector3d& p1, const Eigen::Vector3d& q1, const Eigen::Vector3d& p2, const Eigen::Vector3d& q2,
    ParallelTest parallel = ParallelTest::kAngle) {
  constexpr double kPoint = 1e-24;  // m^2: a segment this short is a point
  const Eigen::Vector3d d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
  const double a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r);
  if (a <= kPoint && e <= kPoint) return {p1, p2};
  if (a <= kPoint) return {p1, p2 + std::clamp(f / e, 0.0, 1.0) * d2};
  const double c = d1.dot(r);
  if (e <= kPoint) return {p1 + std::clamp(-c / a, 0.0, 1.0) * d1, p2};
  const double b = d1.dot(d2);
  const double denom = a * e - b * b;
  const double parallel_below = parallel == ParallelTest::kAngle ? 1e-12 * a * e : 1e-12;
  double s = denom > parallel_below ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
  double t = (b * s + f) / e;
  if (t < 0.0) {
    t = 0.0;
    s = std::clamp(-c / a, 0.0, 1.0);
  } else if (t > 1.0) {
    t = 1.0;
    s = std::clamp((b - c) / a, 0.0, 1.0);
  }
  return {p1 + s * d1, p2 + t * d2};
}

}  // namespace shared_kinematics
