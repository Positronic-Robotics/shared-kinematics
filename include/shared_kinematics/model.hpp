#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "chain.hpp"

namespace shared_kinematics {

// A capsule fixed to a link: the segment [a, b] in link axes, grown by `radius`. a == b gives a sphere.
struct Capsule {
  std::string name;
  int link = 0;
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  double radius = 0.0;
};

// A pad may go past a side by its allowance; a rigid part keeps a margin.
enum class Role { kPad, kRigid };

// A part of the arm that the box and the other bodies bound.
struct Body {
  Capsule capsule;
  Role role = Role::kRigid;
  bool in_planes = true;  // false for a part that sits on the arm mount and cannot leave its place
};

// A pair of capsule groups that keep the self-collision clearance. The indices point into SelfCollision::capsules.
struct SelfPair {
  std::string name;
  std::vector<int> a;
  std::vector<int> b;
};

// The self-collision model: its own capsules, the checked pairs, and the least clearance.
struct SelfCollision {
  std::vector<Capsule> capsules;
  std::vector<SelfPair> pairs;
  double clearance = 0.0;  // m
};

// One arm: its chain, the bounded bodies, the self-collision model, and its base pose in the rig frame.
struct ArmModel {
  Chain chain;
  std::vector<Body> bodies;
  SelfCollision self;
  Eigen::Isometry3d base_pose = Eigen::Isometry3d::Identity();  // T_rig_base

  void validate() const {
    const auto check_capsule = [&](const Capsule& c) {
      if (c.link < 0 || c.link >= chain.link_count())
        throw std::invalid_argument("capsule " + c.name + " names no link of the chain");
      if (!c.a.allFinite() || !c.b.allFinite() || !std::isfinite(c.radius) || c.radius < 0)
        throw std::invalid_argument("capsule " + c.name + " has no finite shape");
    };
    for (const Body& body : bodies) check_capsule(body.capsule);
    for (const Capsule& c : self.capsules) check_capsule(c);
    const int n = static_cast<int>(self.capsules.size());
    for (const SelfPair& pair : self.pairs)
      for (const auto* side : {&pair.a, &pair.b}) {
        if (side->empty()) throw std::invalid_argument("self pair " + pair.name + " has an empty side");
        for (int i : *side)
          if (i < 0 || i >= n) throw std::invalid_argument("self pair " + pair.name + " names no capsule");
      }
    if (!std::isfinite(self.clearance)) throw std::invalid_argument("the self-collision clearance is not finite");
    if (!base_pose.matrix().allFinite()) throw std::invalid_argument("the base pose is not finite");
  }
};

}  // namespace shared_kinematics
