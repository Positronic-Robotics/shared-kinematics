// An FR3 arm model for the C++ tests: the chain of the FR3 URDF, the self-collision capsules and pairs of
// positronic-franka#32, and a small set of bounded bodies with a pad point 0.155 m past the flange.
#pragma once

#include <array>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "shared_kinematics/shared_kinematics.hpp"

namespace fr3 {

using shared_kinematics::ArmModel;
using shared_kinematics::Body;
using shared_kinematics::Capsule;
using shared_kinematics::Chain;
using shared_kinematics::Link;
using shared_kinematics::Role;
using shared_kinematics::SelfPair;

constexpr double kHalfPi = 1.570796326794897;  // as the URDF writes it

inline Eigen::Isometry3d origin(double x, double y, double z, double roll) {
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.translation() = Eigen::Vector3d(x, y, z);
  t.linear() = Eigen::AngleAxisd(roll, Eigen::Vector3d::UnitX()).toRotationMatrix();
  return t;
}

// base, link0 .. link7, link8 (the flange): link k of the FR3 has index k + 1.
inline Chain chain() {
  const Eigen::Vector3d z = Eigen::Vector3d::UnitZ();
  return Chain({
      {"base", -1, Eigen::Isometry3d::Identity(), -1, z},
      {"link0", 0, Eigen::Isometry3d::Identity(), -1, z},
      {"link1", 1, origin(0, 0, 0.333, 0), 0, z},
      {"link2", 2, origin(0, 0, 0, -kHalfPi), 1, z},
      {"link3", 3, origin(0, -0.316, 0, kHalfPi), 2, z},
      {"link4", 4, origin(0.0825, 0, 0, kHalfPi), 3, z},
      {"link5", 5, origin(-0.0825, 0.384, 0, -kHalfPi), 4, z},
      {"link6", 6, origin(0, 0, 0, kHalfPi), 5, z},
      {"link7", 7, origin(0.088, 0, 0, kHalfPi), 6, z},
      {"link8", 8, origin(0, 0, 0.107, 0), -1, z},
  });
}

inline constexpr int kFlange = 9;

// The capsules of positronic-franka#32 (franka_description's *_sc links), on the links of chain().
inline std::vector<Capsule> self_capsules() {
  const Eigen::Vector3d X = Eigen::Vector3d::UnitX(), Z = Eigen::Vector3d::UnitZ();
  const Eigen::Vector3d link7_x(std::cos(M_PI / 4), std::sin(M_PI / 4), 0);
  struct Row {
    int link;
    Eigen::Vector3d center, axis;
    double length, radius;
  };
  const std::vector<Row> rows = {
      {0, {-0.075, 0, 0.06}, X, 0.03, 0.06},  {1, {0, 0, -0.1915}, Z, 0.283, 0.06},
      {2, {0, 0, 0}, Z, 0.12, 0.06},          {3, {0, 0, -0.145}, Z, 0.15, 0.06},
      {4, {0, 0, 0}, Z, 0.12, 0.06},          {5, {0, 0, -0.26}, Z, 0.10, 0.06},
      {5, {0, 0.08, -0.13}, Z, 0.14, 0.025},  {6, {0, 0, -0.03}, Z, 0.08, 0.05},
      {7, {0, 0, 0.01}, Z, 0.14, 0.04},       {7, 0.06 * link7_x + Eigen::Vector3d(0, 0, 0.082), link7_x, 0.01, 0.03},
  };
  std::vector<Capsule> out;
  for (const Row& r : rows)
    out.push_back({"link" + std::to_string(r.link), r.link + 1, r.center - 0.5 * r.length * r.axis,
                   r.center + 0.5 * r.length * r.axis, r.radius});
  return out;
}

inline const std::array<std::pair<int, int>, 8> kPairs = {
    {{0, 5}, {0, 6}, {0, 7}, {1, 5}, {1, 6}, {1, 7}, {2, 5}, {2, 7}}};

inline std::vector<SelfPair> self_pairs(const std::vector<Capsule>& capsules) {
  std::vector<SelfPair> out;
  for (const auto& [la, lb] : kPairs) {
    SelfPair pair{"link" + std::to_string(la) + "-link" + std::to_string(lb), {}, {}};
    for (int i = 0; i < static_cast<int>(capsules.size()); ++i) {
      if (capsules[i].link == la + 1) pair.a.push_back(i);
      if (capsules[i].link == lb + 1) pair.b.push_back(i);
    }
    out.push_back(pair);
  }
  return out;
}

// The pad point: where the closed pads of a 2F-85 meet, 0.155 m along the flange z axis.
inline const Eigen::Vector3d kPadPoint(0, 0, 0.155);

// The self capsules of links 2 .. 7 as rigid bodies, a gripper body, and the pad point.
inline std::vector<Body> bodies() {
  std::vector<Body> out;
  for (const Capsule& c : self_capsules()) out.push_back({c, Role::kRigid, c.link > 2});
  out.push_back({{"gripper", kFlange, {0, 0, 0.01}, {0, 0, 0.11}, 0.045}, Role::kRigid, true});
  out.push_back({{"pads", kFlange, kPadPoint, kPadPoint, 0.0}, Role::kPad, true});
  return out;
}

inline ArmModel model(const Eigen::Isometry3d& base_pose = Eigen::Isometry3d::Identity()) {
  ArmModel m;
  m.chain = chain();
  m.self.capsules = self_capsules();
  m.self.pairs = self_pairs(m.self.capsules);
  m.self.clearance = 0.09;
  m.bodies = bodies();
  m.base_pose = base_pose;
  return m;
}

inline const std::array<double, 7> kLower = {-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159};
inline const std::array<double, 7> kUpper = {2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159};

inline Eigen::VectorXd random_q(std::mt19937& rng) {
  std::uniform_real_distribution<double> u(0.0, 1.0);
  Eigen::VectorXd q(7);
  for (int i = 0; i < 7; ++i) q(i) = kLower[i] + u(rng) * (kUpper[i] - kLower[i]);
  return q;
}

// A pose with the flange down and the pad point at (0.55, 0, 0.14) m.
inline Eigen::VectorXd low() {
  Eigen::VectorXd q(7);
  q << 0.0, 0.2, 0.0, -2.2, 0.0, 2.4, M_PI / 4;
  return q;
}

}  // namespace fr3
