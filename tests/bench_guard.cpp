// Times clip and speed_scale on the FR3 model of tests/fr3_fixture.hpp, with a five-sided box and without one.

#include <algorithm>
#include <chrono>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "fr3_fixture.hpp"
#include "shared_kinematics/shared_kinematics.hpp"

namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;
using shared_kinematics::Guard;
using shared_kinematics::Plane;
using shared_kinematics::SideKind;

Plane side(const std::string& name, SideKind kind, const Vector3d& n, const Vector3d& through, double pad) {
  return {name, kind, n.normalized(), -n.normalized().dot(through), pad, -0.005};
}

// A desk 5 cm below the pad at fr3::low(), walls 0.4 m to the front and the sides, a virtual back side.
std::vector<Plane> box() {
  return {side("desk", SideKind::kSurface, {0, 0, 1}, {0, 0, 0.09}, 0.01),
          side("front", SideKind::kSurface, {-1, 0, 0}, {0.95, 0, 0}, 0.0),
          side("left", SideKind::kSurface, {0, -1, 0}, {0, 0.6, 0}, 0.0),
          side("right", SideKind::kSurface, {0, 1, 0}, {0, -0.6, 0}, 0.0),
          side("back", SideKind::kVirtual, {1, 0, 0}, {-0.35, 0, 0}, 0.0)};
}

void report(const std::string& what, std::vector<double> us) {
  std::sort(us.begin(), us.end());
  const auto at = [&](double p) { return us[static_cast<std::size_t>(p * (us.size() - 1))]; };
  std::cout << std::left << std::setw(58) << what << std::right << std::fixed << std::setprecision(1)
            << " median " << std::setw(8) << at(0.5) << " us, p99 " << std::setw(8) << at(0.99) << " us, max "
            << std::setw(8) << us.back() << " us (" << us.size() << " calls)\n";
}

std::vector<double> time_each(int n, const std::function<void(int)>& call) {
  std::vector<double> us;
  for (int i = 0; i < n; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    call(i);
    us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
  }
  return us;
}

}  // namespace

int main() {
  const Guard bare(fr3::model()), boxed(fr3::model(), box());
  std::mt19937 rng(7);
  std::normal_distribution<double> n(0.0, 1.0);
  const int kCalls = 2000;

  // Policy-rate targets: 15 Hz at up to 0.3 rad/s per joint, from postures around fr3::low().
  std::vector<VectorXd> from, to;
  for (int i = 0; i < kCalls; ++i) {
    VectorXd q = fr3::low(), dq(7);
    for (int j = 0; j < 7; ++j) {
      q(j) += 0.3 * n(rng);
      dq(j) = 0.02 * n(rng);
    }
    from.push_back(q);
    to.push_back(q + dq);
  }
  report("clip, policy-rate target, no box", time_each(kCalls, [&](int i) { bare.clip(from[i], to[i]); }));
  report("clip, policy-rate target, box of 5 sides", time_each(kCalls, [&](int i) { boxed.clip(from[i], to[i]); }));

  // Park moves: from a working posture to fr3::low() and back, 1-2 rad in all.
  std::vector<VectorXd> park_from;
  for (int i = 0; i < 200; ++i) {
    VectorXd q = fr3::low();
    for (int j = 0; j < 7; ++j) q(j) += 0.35 * n(rng);
    park_from.push_back(q);
  }
  VectorXd park(7);
  park << 0.0, -0.78, 0.0, -2.36, 0.0, 1.57, 0.78;
  report("clip, park move, no box", time_each(200, [&](int i) { bare.clip(park_from[i], park); }));
  report("clip, park move, box of 5 sides", time_each(200, [&](int i) { boxed.clip(park_from[i], park); }));

  // One 1 kHz tick at up to 0.5 rad/s per joint.
  std::vector<VectorXd> tick_q, tick_dq;
  for (int i = 0; i < kCalls * 5; ++i) {
    VectorXd q = fr3::low(), dq(7);
    for (int j = 0; j < 7; ++j) {
      q(j) += 0.3 * n(rng);
      dq(j) = 0.0005 * n(rng);
    }
    tick_q.push_back(q);
    tick_dq.push_back(dq);
  }
  report("speed_scale, one tick, no box", time_each(kCalls * 5, [&](int i) {
           bare.speed_scale(tick_q[i], tick_dq[i], 0.001);
         }));
  report("speed_scale, one tick, box of 5 sides", time_each(kCalls * 5, [&](int i) {
           boxed.speed_scale(tick_q[i], tick_dq[i], 0.001);
         }));

  // A second arm 1.4 m away: its capsules now, and its committed move.
  Eigen::Isometry3d b_pose = Eigen::Isometry3d::Identity();
  b_pose.translation() = Vector3d(1.4, 0, 0);
  b_pose.linear() = Eigen::AngleAxisd(M_PI, Vector3d::UnitZ()).toRotationMatrix();
  const Guard other(fr3::model(b_pose));
  VectorXd b_to = fr3::low();
  b_to(1) += 0.2;
  const auto b_now = other.capsules_at(fr3::low());
  const auto b_move = other.swept_capsules(fr3::low(), b_to);
  std::cout << "other arm: " << b_now.size() << " capsules now, " << b_move.size() << " in its committed move\n";
  report("clip, policy-rate target, box and the other arm's move",
         time_each(kCalls, [&](int i) { boxed.clip(from[i], to[i], {b_move}); }));
  report("speed_scale, one tick, box and the other arm", time_each(kCalls * 5, [&](int i) {
           boxed.speed_scale(tick_q[i], tick_dq[i], 0.001, {b_now});
         }));
  return 0;
}
