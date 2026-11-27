// The FR3 self-collision walk of positronic-franka 2d8c210 (#32), the reference of the parity tests.
// Copied from src/self_collision.hpp and src/pose_ik.hpp of that commit (Apache License 2.0).
#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace positronic_franka {

using Vector7d = Eigen::Matrix<double, 7, 1>;

// The FR3 self-collision capsules of franka_description (robots/common/franka_arm.xacro, the *_sc links).
// The gripper on the flange has no capsule.
struct Capsule {
  int link;                // 0 is the base, 7 is link7
  Eigen::Vector3d center;  // in link axes, m
  Eigen::Vector3d axis;    // unit, in link axes
  double length;           // between the two sphere centres, m
  double radius;           // m
};

// link7's capsules are in axes turned pi/4 about its z axis.
inline std::array<Capsule, 10> fr3_self_collision_capsules() {
  const Eigen::Vector3d X = Eigen::Vector3d::UnitX(), Z = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d link7_x(std::cos(M_PI / 4), std::sin(M_PI / 4), 0);
  return {{
      {0, {-0.075, 0, 0.06}, X, 0.03, 0.06},
      {1, {0, 0, -0.1915}, Z, 0.283, 0.06},
      {2, {0, 0, 0}, Z, 0.12, 0.06},
      {3, {0, 0, -0.145}, Z, 0.15, 0.06},
      {4, {0, 0, 0}, Z, 0.12, 0.06},
      {5, {0, 0, -0.26}, Z, 0.10, 0.06},
      {5, {0, 0.08, -0.13}, Z, 0.14, 0.025},
      {6, {0, 0, -0.03}, Z, 0.08, 0.05},
      {7, {0, 0, 0.01}, Z, 0.14, 0.04},
      {7, 0.06 * link7_x + Eigen::Vector3d(0, 0, 0.082), link7_x, 0.01, 0.03},
  }};
}

// The arm link pairs that franka_description's SRDF checks.
constexpr std::array<std::pair<int, int>, 8> FR3_SELF_COLLISION_PAIRS = {
    {{0, 5}, {0, 6}, {0, 7}, {1, 5}, {1, 6}, {1, 7}, {2, 5}, {2, 7}}};

// The least clearance between the capsules above that an IK answer keeps. The FR3 self-collision reflex
// stopped moves at 0.064 m and less, and let moves through at 0.07 m and more.
constexpr double IK_SELF_COLLISION_CLEARANCE = 0.09;

// How far below its floor the clearance may fall between two checked points of a move.
constexpr double IK_SELF_COLLISION_PATH_TOLERANCE = 0.001;

// Franka's modified DH parameters of joints 1 .. 7 and of the flange.
constexpr std::array<double, 8> FR3_DH_A = {0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0};
constexpr std::array<double, 8> FR3_DH_D = {0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107};
constexpr std::array<double, 8> FR3_DH_ALPHA = {0, -M_PI / 2, M_PI / 2, M_PI / 2, -M_PI / 2, M_PI / 2, M_PI / 2, 0};

// Base-frame transforms of link0 .. link7 and of the flange (link8).
using LinkFrames = std::array<Eigen::Isometry3d, 9>;

inline LinkFrames fr3_link_frames(const Vector7d& q) {
  LinkFrames f;
  f[0] = Eigen::Isometry3d::Identity();
  for (int i = 0; i < 8; ++i)
    f[i + 1] = f[i] * Eigen::AngleAxisd(FR3_DH_ALPHA[i], Eigen::Vector3d::UnitX()) *
               Eigen::Translation3d(FR3_DH_A[i], 0, 0) * Eigen::AngleAxisd(i < 7 ? q(i) : 0.0, Eigen::Vector3d::UnitZ()) *
               Eigen::Translation3d(0, 0, FR3_DH_D[i]);
  return f;
}

// The closest points of segments [p1, q1] and [p2, q2] (Ericson, Real-Time Collision Detection, 5.1.9).
inline std::pair<Eigen::Vector3d, Eigen::Vector3d> closest_points_of_segments(const Eigen::Vector3d& p1,
                                                                              const Eigen::Vector3d& q1,
                                                                              const Eigen::Vector3d& p2,
                                                                              const Eigen::Vector3d& q2) {
  const Eigen::Vector3d d1 = q1 - p1, d2 = q2 - p2, r = p1 - p2;
  const double a = d1.dot(d1), e = d2.dot(d2), f = d2.dot(r), c = d1.dot(r), b = d1.dot(d2);
  const double denom = a * e - b * b;
  double s = denom > 1e-12 ? std::clamp((b * f - c * e) / denom, 0.0, 1.0) : 0.0;
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

struct SelfCollisionClearance {
  double clearance;  // m; negative when the capsules overlap
  Vector7d gradient;  // d clearance / dq
};

// How fast a point fixed to `link` moves along `direction`, per radian of each joint.
inline Vector7d point_speed_along(const LinkFrames& f, int link, const Eigen::Vector3d& p,
                                  const Eigen::Vector3d& direction) {
  Vector7d g = Vector7d::Zero();
  for (int j = 0; j < std::min(link, 7); ++j) {
    const Eigen::Vector3d axis = f[j + 1].linear().col(2);
    g(j) = direction.dot(axis.cross(p - f[j + 1].translation()));
  }
  return g;
}

using CapsuleSegment = std::pair<Eigen::Vector3d, Eigen::Vector3d>;  // the two sphere centres, base frame

// The clearance of two capsules, and its gradient in the joints.
inline SelfCollisionClearance capsule_clearance(const LinkFrames& f, const Capsule& a, const CapsuleSegment& sa,
                                                const Capsule& b, const CapsuleSegment& sb) {
  const auto [pa, pb] = closest_points_of_segments(sa.first, sa.second, sb.first, sb.second);
  const double distance = (pa - pb).norm();
  // Crossed axes give no direction to push apart; the IK keeps 0.09 m, far from that depth.
  const Eigen::Vector3d n = distance > 1e-9 ? Eigen::Vector3d((pa - pb) / distance) : Eigen::Vector3d::Zero();
  return {distance - a.radius - b.radius, point_speed_along(f, a.link, pa, n) - point_speed_along(f, b.link, pb, n)};
}

// The clearance of each pair in FR3_SELF_COLLISION_PAIRS, over the closest of their capsules.
inline std::array<SelfCollisionClearance, FR3_SELF_COLLISION_PAIRS.size()> self_collision_clearances(
    const Vector7d& q) {
  const LinkFrames f = fr3_link_frames(q);
  const auto capsules = fr3_self_collision_capsules();
  std::array<CapsuleSegment, capsules.size()> segments;
  for (std::size_t i = 0; i < capsules.size(); ++i) {
    const Capsule& cap = capsules[i];
    const Eigen::Vector3d center = f[cap.link] * cap.center;
    const Eigen::Vector3d half = f[cap.link].linear() * cap.axis * (0.5 * cap.length);
    segments[i] = {center - half, center + half};
  }
  std::array<SelfCollisionClearance, FR3_SELF_COLLISION_PAIRS.size()> out;
  for (std::size_t k = 0; k < FR3_SELF_COLLISION_PAIRS.size(); ++k) {
    const auto [la, lb] = FR3_SELF_COLLISION_PAIRS[k];
    out[k] = {std::numeric_limits<double>::infinity(), Vector7d::Zero()};
    for (std::size_t i = 0; i < capsules.size(); ++i)
      for (std::size_t j = 0; j < capsules.size(); ++j)
        if (capsules[i].link == la && capsules[j].link == lb) {
          const SelfCollisionClearance c = capsule_clearance(f, capsules[i], segments[i], capsules[j], segments[j]);
          if (c.clearance < out[k].clearance) out[k] = c;
        }
  }
  return out;
}

// For each joint, the most that the clearance of a checked pair changes per radian of that joint, at any q.
// A joint turns a point at most its distance from the joint origin per radian. The DH chain from the joint
// origin to the link origin, plus the farthest capsule end from the link origin, bounds that distance.
// A joint before both links of a pair turns the two links together and does not change their clearance.
inline Vector7d fr3_clearance_rate_bounds() {
  Vector7d bound = Vector7d::Zero();
  for (const auto& [la, lb] : FR3_SELF_COLLISION_PAIRS)
    for (const Capsule& cap : fr3_self_collision_capsules()) {
      if (cap.link != lb) continue;
      const Eigen::Vector3d half = cap.axis * (0.5 * cap.length);
      double reach = std::max((cap.center - half).norm(), (cap.center + half).norm());
      for (int j = lb - 1; j >= la; --j) {  // joint j has its origin at link j + 1
        bound(j) = std::max(bound(j), reach);
        reach += std::hypot(FR3_DH_A[j], FR3_DH_D[j]);  // from link j + 1 to link j
      }
    }
  return bound;
}

// The point of the move from `from` to `to` farthest along it whose path keeps each checked pair at its floor:
// IK_SELF_COLLISION_CLEARANCE, or its own clearance at `from` when that is less.
// Between two checked points a pair's clearance falls at most IK_SELF_COLLISION_PATH_TOLERANCE below its floor:
// the points are close enough for the rate bound of each joint (fr3_clearance_rate_bounds).
inline Vector7d stop_before_self_collision(const Vector7d& from, const Vector7d& to) {
  constexpr int kLeastSamples = 16;
  constexpr int kBisections = 8;
  static const Vector7d rate_bounds = fr3_clearance_rate_bounds();
  // The most a pair's clearance changes over the whole move. Between two points 1 / samples apart that both keep
  // its floor, the clearance falls at most change_over_move / (2 * samples) below that floor.
  const double change_over_move = rate_bounds.dot((to - from).cwiseAbs());
  const int samples =
      std::max(kLeastSamples, static_cast<int>(std::ceil(change_over_move / (2 * IK_SELF_COLLISION_PATH_TOLERANCE))));
  std::array<double, FR3_SELF_COLLISION_PAIRS.size()> floors;
  const auto at_from = self_collision_clearances(from);
  for (std::size_t k = 0; k < floors.size(); ++k)
    floors[k] = std::min(IK_SELF_COLLISION_CLEARANCE, at_from[k].clearance);
  const auto clear = [&](double s) {
    const auto pairs = self_collision_clearances(from + s * (to - from));
    for (std::size_t k = 0; k < pairs.size(); ++k)
      if (pairs[k].clearance < floors[k]) return false;
    return true;
  };
  double safe = 0.0;
  for (int i = 1; i <= samples; ++i) {
    const double s = static_cast<double>(i) / samples;
    if (clear(s)) {
      safe = s;
      continue;
    }
    double blocked = s;
    for (int b = 0; b < kBisections; ++b) {
      const double mid = 0.5 * (safe + blocked);
      (clear(mid) ? safe : blocked) = mid;
    }
    return from + safe * (to - from);
  }
  return to;
}

}  // namespace positronic_franka
