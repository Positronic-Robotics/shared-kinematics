// Checks the guard in include/shared_kinematics on the FR3 model of tests/fr3_fixture.hpp. Needs only Eigen.
// SK_PARITY_TRIPLES names an optional file of recorded walks of positronic-franka#32 (README, "Parity").

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "fr3_fixture.hpp"
#include "reference/positronic_franka_32.hpp"
#include "shared_kinematics/shared_kinematics.hpp"

namespace {

using Eigen::Vector3d;
using Eigen::VectorXd;
using shared_kinematics::ClipResult;
using shared_kinematics::Guard;
using shared_kinematics::GuardParams;
using shared_kinematics::OtherBody;
using shared_kinematics::Plane;
using shared_kinematics::SideKind;

struct Failure {
  std::string what;
};

struct Skip {};

void check(bool ok, const std::string& what) {
  if (!ok) throw Failure{what};
}

std::string str(double x) {
  std::ostringstream out;
  out << x;
  return out.str();
}

positronic_franka::Vector7d v7(const VectorXd& q) { return q; }

const Guard& bare_guard() {
  static const Guard guard(fr3::model());
  return guard;
}

// The rig-frame pad point at q.
Vector3d pad_at(const Guard& guard, const VectorXd& q) {
  return guard.model().base_pose * (guard.model().chain.frames(q)[fr3::kFlange] * fr3::kPadPoint);
}

// The least distance past its floor of any bounded part at q, over the planes of the guard.
double least_to_floor(const Guard& guard, const VectorXd& q, const std::string& side = "") {
  double least = std::numeric_limits<double>::infinity();
  for (const auto& row : guard.distances(q).planes)
    if (side.empty() || row.side == side) least = std::min(least, row.distance - row.floor);
  return least;
}

// The least, over the plane checks and the points of the move, of the distance to the floor of the clip: the
// floor of the box, or the distance at `from` when that is less.
double least_to_floor_on_the_move(const Guard& guard, const VectorXd& from, const VectorXd& to) {
  const auto at_from = guard.distances(from).planes;
  double least = std::numeric_limits<double>::infinity();
  for (int i = 0; i <= 2000; ++i) {
    const auto rows = guard.distances(from + i / 2000.0 * (to - from)).planes;
    for (std::size_t k = 0; k < rows.size(); ++k)
      least = std::min(least, rows[k].distance - std::min(rows[k].floor, at_from[k].distance));
  }
  return least;
}

// Joints that move the pad by `dx` in the base frame to first order, from q, plus `extra`.
VectorXd move_pad(const Guard& guard, const VectorXd& q, const Vector3d& dx, const VectorXd& extra) {
  const auto f = guard.model().chain.frames(q);
  const Vector3d p = f[fr3::kFlange] * fr3::kPadPoint;
  Eigen::Matrix<double, 3, 7> J;
  for (int j = 0; j < 7; ++j) {
    VectorXd e = VectorXd::Zero(7);
    e(j) = 1.0;
    J.col(j) = guard.model().chain.point_velocity(f, fr3::kFlange, p, e);
  }
  return q + J.transpose() * (J * J.transpose()).ldlt().solve(dx) + extra;
}

Plane plane(const std::string& name, const Vector3d& n, const Vector3d& through, double pad_allowance = 0.0,
            double margin = 0.005, SideKind kind = SideKind::kSurface) {
  return {name, kind, n.normalized(), -n.normalized().dot(through), pad_allowance, -margin};
}

// ---- Kinematics ------------------------------------------------------------------------------------------

void the_chain_frames_match_the_fr3_dh_frames_within_0_5_mm() {
  std::mt19937 rng(1);
  double worst_mm = 0.0, worst_rad = 0.0;
  for (int trial = 0; trial < 1000; ++trial) {
    const VectorXd q = fr3::random_q(rng);
    const auto dh = positronic_franka::fr3_link_frames(v7(q));
    const auto ours = fr3::chain().frames(q);
    for (int k = 0; k < 9; ++k) {
      worst_mm = std::max(worst_mm, (dh[k].translation() - ours[k + 1].translation()).norm() * 1e3);
      worst_rad = std::max(worst_rad, Eigen::AngleAxisd(dh[k].linear().transpose() * ours[k + 1].linear()).angle());
    }
  }
  std::cout << "  FK miss against the DH frames: " << worst_mm << " mm, " << worst_rad << " rad\n";
  check(worst_mm < 0.5 && worst_rad < 1e-6, "the chain misses the DH frames");
}

void the_point_velocity_is_the_derivative_of_the_point_position() {
  std::mt19937 rng(2);
  const auto chain = fr3::chain();
  for (int trial = 0; trial < 50; ++trial) {
    const VectorXd q = fr3::random_q(rng);
    const auto f = chain.frames(q);
    const Vector3d local(0.01, -0.02, 0.15);
    const Vector3d p = f[fr3::kFlange] * local;
    for (int j = 0; j < 7; ++j) {
      VectorXd e = VectorXd::Zero(7);
      e(j) = 1.0;
      const double h = 1e-6;
      const Vector3d numeric =
          (chain.frames(q + h * e)[fr3::kFlange] * local - chain.frames(q - h * e)[fr3::kFlange] * local) / (2 * h);
      check((chain.point_velocity(f, fr3::kFlange, p, e) - numeric).norm() < 1e-7, "the point velocity is wrong");
    }
  }
}

void the_self_clearances_and_rate_bounds_equal_those_of_32() {
  const auto reference = positronic_franka::fr3_clearance_rate_bounds();
  check((bare_guard().self_rate_bounds() - VectorXd(reference)).cwiseAbs().maxCoeff() < 1e-12,
        "the self rate bounds differ from #32");
  std::mt19937 rng(3);
  for (int trial = 0; trial < 500; ++trial) {
    const VectorXd q = fr3::random_q(rng);
    const auto ours = bare_guard().distances(q).pairs;
    const auto theirs = positronic_franka::self_collision_clearances(v7(q));
    check(ours.size() == theirs.size(), "the pair count differs from #32");
    for (std::size_t k = 0; k < ours.size(); ++k) {
      check(std::abs(ours[k].clearance - theirs[k].clearance) < 1e-12, "pair " + ours[k].pair + " differs");
      check((ours[k].gradient - VectorXd(theirs[k].gradient)).cwiseAbs().maxCoeff() < 1e-9,
            "the gradient of pair " + ours[k].pair + " differs");
    }
  }
}

// ---- Parity with positronic-franka#32 ---------------------------------------------------------------------

struct Parity {
  int moves = 0, stopped = 0, equal = 0;
  double worst = 0.0;
};

void compare(Parity& p, const VectorXd& from, const VectorXd& to) {
  const VectorXd theirs = positronic_franka::stop_before_self_collision(v7(from), v7(to));
  const ClipResult ours = bare_guard().clip(from, to);
  ++p.moves;
  p.stopped += theirs != v7(to) ? 1 : 0;
  p.equal += ours.q == theirs ? 1 : 0;
  p.worst = std::max(p.worst, (ours.q - theirs).cwiseAbs().maxCoeff());
  check(ours.bound_sides.empty() && ours.others_bound.empty() && !ours.slid, "a move without a box bound a side");
  check(ours.self_bound == (ours.q != to), "self_bound does not say whether the move stopped");
}

void with_no_box_clip_equals_the_walk_of_32_on_synthetic_moves() {
  std::mt19937 rng(4);
  std::normal_distribution<double> n(0.0, 1.0);
  Parity p;
  for (double scale : {0.02, 0.1, 0.5}) {
    for (int trial = 0; trial < 400; ++trial) {
      const VectorXd from = fr3::random_q(rng);
      VectorXd step(7);
      for (int i = 0; i < 7; ++i) step(i) = scale * n(rng);
      compare(p, from, from + step);
    }
  }
  for (int trial = 0; trial < 400; ++trial) compare(p, fr3::random_q(rng), fr3::random_q(rng));
  // The moves of the #32 tests.
  VectorXd a(7), b(7), folded(7), out(7), seven_to_two(7);
  a << -2.025573, 1.055818, 2.246552, -2.790485, 2.673202, 2.095729, -1.071398;
  b << 1.167132, 0.990198, -2.123749, -2.129653, 0.255810, 1.355959, 2.292034;
  folded << 0.98877418, -1.15661502, -0.96312511, -2.93868756, -0.990777671, 2.05664039, 1.14565432;
  out << 1.173, -1.186, -1.034, -2.834, -0.818, 2.141, 1.153;
  seven_to_two << 1.384, -1.477, -0.791, -3.032, -0.499, 1.563, 1.542;
  for (const auto& [from, to] : std::vector<std::pair<VectorXd, VectorXd>>{
           {a, b}, {folded, out}, {folded, VectorXd(folded - (out - folded))}, {folded, seven_to_two}})
    compare(p, from, to);
  // Moves around the folded elbow, where pairs start inside the clearance.
  for (int trial = 0; trial < 400; ++trial) {
    VectorXd step(7);
    for (int i = 0; i < 7; ++i) step(i) = 0.2 * n(rng);
    compare(p, folded, folded + step);
  }
  std::cout << "  " << p.moves << " moves, " << p.stopped << " stopped by #32, " << p.equal
            << " bit-equal, worst joint difference " << p.worst << " rad\n";
  check(p.stopped > 100, "too few moves stop to test the stop");
  check(p.worst < 1e-9, "a clip differs from the walk of #32 by " + str(p.worst) + " rad");
}

// Each line: a label, then from[7], to[7], and the result of #32's walk[7] in a recorded replay.
void with_no_box_clip_equals_the_walk_of_32_on_the_recorded_replay() {
  const char* path = std::getenv("SK_PARITY_TRIPLES");
  if (path == nullptr) throw Skip{};
  std::ifstream in(path);
  check(in.good(), std::string("cannot read ") + path);
  Parity p;
  int recorded_equal = 0;
  double recorded_worst = 0.0;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream row(line);
    std::string label;
    row >> label;
    VectorXd from(7), to(7), result(7);
    for (VectorXd* v : {&from, &to, &result})
      for (int i = 0; i < 7; ++i) row >> (*v)(i);
    check(!row.fail(), "a bad line in " + std::string(path));
    const VectorXd ours = bare_guard().clip(from, to).q;
    recorded_equal += ours == result ? 1 : 0;
    recorded_worst = std::max(recorded_worst, (ours - result).cwiseAbs().maxCoeff());
    compare(p, from, to);
  }
  std::cout << "  " << p.moves << " recorded moves, " << p.stopped << " stopped by #32; against the recorded result: "
            << recorded_equal << " bit-equal, worst " << recorded_worst << " rad; against the reference: " << p.equal
            << " bit-equal, worst " << p.worst << " rad\n";
  check(p.moves > 0, "the file holds no moves");
  check(recorded_worst < 1e-9 && p.worst < 1e-9, "a clip differs from #32 on the replay");
}

// ---- No-op ------------------------------------------------------------------------------------------------

void with_no_box_and_no_other_body_speed_scale_is_one_and_clip_keeps_the_self_pairs_only() {
  std::mt19937 rng(5);
  std::normal_distribution<double> n(0.0, 0.01);
  for (int trial = 0; trial < 200; ++trial) {
    const VectorXd q = fr3::random_q(rng);
    VectorXd dq(7);
    for (int i = 0; i < 7; ++i) dq(i) = n(rng);
    const auto scale = bare_guard().speed_scale(q, dq, 0.001);
    check(scale.s == 1.0 && !scale.slowest, "speed_scale is not 1 without a box");
    check(bare_guard().distances(q).planes.empty(), "a guard without a box reports a plane");
  }
  shared_kinematics::ArmModel bare = fr3::model();
  bare.self.pairs.clear();
  const Guard nothing(bare);
  for (int trial = 0; trial < 200; ++trial) {
    const VectorXd from = fr3::random_q(rng), to = fr3::random_q(rng);
    const ClipResult r = nothing.clip(from, to);
    check(r.q == to && r.fraction == 1.0 && !r.self_bound, "a guard with nothing to check changed a target");
  }
}

// ---- Planes -----------------------------------------------------------------------------------------------

// A wall 8 cm in front of the pad, and a target that takes the pad 6 cm through it.
void a_target_past_a_wall_stops_within_1_mm_of_the_floor_in_true_fk() {
  const VectorXd low = fr3::low();
  const Vector3d pad = pad_at(bare_guard(), low);
  const Guard guard(fr3::model(), {plane("front", Vector3d(-1, 0, 0), pad + Vector3d(0.08, 0, 0))});
  const VectorXd to = move_pad(guard, low, Vector3d(0.14, 0, 0), VectorXd::Zero(7));
  check(least_to_floor(guard, to) < -0.05, "the target is not past the wall");
  const ClipResult r = guard.clip(low, to);
  check(r.bound_sides == std::vector<std::string>{"front"}, "the front wall does not bind");
  const VectorXd stop = low + r.fraction * (to - low);
  const double at_stop = least_to_floor(guard, stop, "front");
  const double on_path = least_to_floor_on_the_move(guard, low, r.q);
  std::cout << "  at the stop " << at_stop * 1e3 << " mm from the floor; least on the committed move "
            << on_path * 1e3 << " mm\n";
  check(at_stop >= 0 && at_stop <= 0.001, "the stop is more than 1 mm from the floor");
  check(on_path >= -0.001, "the committed move goes more than 1 mm past the floor");
  check(r.fraction > 0.1 && r.fraction < 0.5, "the move stops at " + str(r.fraction) + ", not near 0.25");
}

// The pad starts 2 cm below a desk: a move further down stops at once, a move up plays to its end.
void a_point_past_a_floor_can_move_back_in_only() {
  const VectorXd low = fr3::low();
  const Vector3d pad = pad_at(bare_guard(), low);
  const Guard guard(fr3::model(), {plane("desk", Vector3d(0, 0, 1), pad + Vector3d(0, 0, 0.02))});
  check(least_to_floor(guard, low, "desk") < -0.015, "the pad does not start past the desk");
  const VectorXd down = move_pad(guard, low, Vector3d(0, 0, -0.05), VectorXd::Zero(7));
  const ClipResult r = guard.clip(low, down);
  check(least_to_floor_on_the_move(guard, low, r.q) >= least_to_floor(guard, low) - 0.001,
        "the pad goes further past the desk");
  check(pad_at(guard, r.q).z() >= pad_at(guard, low).z() - 0.001, "the pad moves down");
  const VectorXd up = move_pad(guard, low, Vector3d(0, 0, 0.05), VectorXd::Zero(7));
  check(guard.clip(low, up).q == up, "the move back in stops");
}

// A target down through the desk and to the side: the move stops at the desk and slides along it.
void the_slide_keeps_the_motion_along_a_side() {
  const VectorXd low = fr3::low();
  const Vector3d pad = pad_at(bare_guard(), low);
  const Guard guard(fr3::model(), {plane("desk", Vector3d(0, 0, 1), pad - Vector3d(0, 0, 0.03))});
  const VectorXd to = move_pad(guard, low, Vector3d(0, 0.10, -0.08), VectorXd::Zero(7));
  const ClipResult r = guard.clip(low, to);
  check(r.bound_sides == std::vector<std::string>{"desk"} && r.slid, "the move does not slide along the desk");
  const Vector3d stop = pad_at(guard, low + r.fraction * (to - low)), end = pad_at(guard, r.q);
  std::cout << "  the pad stops after " << (stop - pad).y() * 1e3 << " mm to the side and slides to "
            << (end - pad).y() * 1e3 << " mm; height above the desk " << (end.z() - pad.z() + 0.03) * 1e3 << " mm\n";
  check((end - pad).y() > 0.08, "the slide does not take the pad most of the way to the side");
  check((end - pad).y() > (stop - pad).y() + 0.04, "the slide adds little to the stop");
  check(least_to_floor_on_the_move(guard, low, r.q) >= -0.001, "the committed move goes past the desk");
}

void a_corner_of_two_sides_binds_both() {
  const VectorXd low = fr3::low();
  const Vector3d pad = pad_at(bare_guard(), low);
  const Guard guard(fr3::model(), {plane("desk", Vector3d(0, 0, 1), pad - Vector3d(0, 0, 0.03)),
                                   plane("front", Vector3d(-1, 0, 0), pad + Vector3d(0.03, 0, 0))});
  const VectorXd to = move_pad(guard, low, Vector3d(0.08, 0, -0.08), VectorXd::Zero(7));
  const ClipResult r = guard.clip(low, to);
  std::vector<std::string> sides = r.bound_sides;
  std::sort(sides.begin(), sides.end());
  check(sides == std::vector<std::string>{"desk", "front"}, "the corner does not bind both sides");
  check(least_to_floor_on_the_move(guard, low, r.q) >= -0.001, "the committed move goes past a side");
}

// ---- Two arms ---------------------------------------------------------------------------------------------

// Arm B faces arm A 1.4 m away and reaches 10 cm toward it. Arm A reaches 30 cm toward B.
void a_move_that_crosses_the_other_arms_committed_move_stops_before_it() {
  Eigen::Isometry3d b_pose = Eigen::Isometry3d::Identity();
  b_pose.translation() = Vector3d(1.4, 0, 0);
  b_pose.linear() = Eigen::AngleAxisd(M_PI, Vector3d::UnitZ()).toRotationMatrix();
  const Guard a(fr3::model()), b(fr3::model(b_pose));
  const VectorXd low = fr3::low();
  const VectorXd a_to = move_pad(a, low, Vector3d(0.30, 0, 0), VectorXd::Zero(7));
  const VectorXd b_to = move_pad(b, low, Vector3d(0.10, 0, 0), VectorXd::Zero(7));
  const OtherBody b_move = b.swept_capsules(low, b_to);
  const double clearance = a.params().other_clearance;
  const auto least_clearance = [&](const VectorXd& from, const VectorXd& to) {
    double least = std::numeric_limits<double>::infinity();
    for (int i = 0; i <= 200; ++i) {
      const VectorXd q = from + i / 200.0 * (to - from);
      for (int k = 0; k <= 50; ++k) {
        const OtherBody other = b.capsules_at(low + k / 50.0 * (b_to - low));
        for (const auto& row : a.distances(q, {other}).others) least = std::min(least, row.clearance);
      }
    }
    return least;
  };
  check(least_clearance(low, a_to) < clearance - 0.05, "the two moves do not cross");
  const ClipResult r = a.clip(low, a_to, {b_move});
  check(r.others_bound == std::vector<int>{0}, "the other arm does not bind the move");
  const double kept = least_clearance(low, r.q);
  std::cout << "  least clearance to the other arm's move " << kept * 1e3 << " mm; the move stops at "
            << r.fraction << "\n";
  check(kept >= clearance - 0.001, "the move comes inside the clearance to the other arm");
  check(r.fraction > 0.1, "the move stops far before the other arm");
}

// ---- speed_scale ------------------------------------------------------------------------------------------

void speed_scale_is_one_far_from_all_sides() {
  const Guard guard(fr3::model(), {plane("desk", Vector3d(0, 0, 1), Vector3d(0, 0, -0.5)),
                                   plane("front", Vector3d(-1, 0, 0), Vector3d(2.0, 0, 0))});
  std::mt19937 rng(6);
  std::normal_distribution<double> n(0.0, 1e-4);
  for (int trial = 0; trial < 100; ++trial) {
    VectorXd dq(7);
    for (int i = 0; i < 7; ++i) dq(i) = n(rng);
    const auto s = guard.speed_scale(fr3::low(), dq, 0.001);
    check(s.s == 1.0 && !s.slowest, "speed_scale is not 1 far from the sides");
  }
}

// The pad 8 cm from a wall moves toward it at 0.5 m/s: the scale holds it to k times its distance to the floor.
void speed_scale_keeps_the_step_direction_and_obeys_a3() {
  const VectorXd low = fr3::low();
  const Vector3d pad = pad_at(bare_guard(), low);
  const double margin = 0.005;
  const Guard guard(fr3::model(), {plane("front", Vector3d(-1, 0, 0), pad + Vector3d(0.08, 0, 0), 0.0, margin)});
  const double dt = 0.001;
  const VectorXd dq = move_pad(guard, low, Vector3d(0.5 * dt, 0, 0), VectorXd::Zero(7)) - low;
  const auto s = guard.speed_scale(low, dq, dt);
  check(s.slowest == std::optional<std::string>("front") && s.s < 1.0, "the wall does not slow the step");
  // The fastest approach of any part, scaled, against k times its distance to the floor.
  const auto f = guard.model().chain.frames(low);
  double worst = 0.0;
  for (const auto& body : guard.model().bodies) {
    if (!body.in_planes) continue;
    for (const Vector3d& end : {body.capsule.a, body.capsule.b}) {
      const Vector3d p = f[body.capsule.link] * end;
      const double floor = body.role == shared_kinematics::Role::kPad ? 0.0 : margin;
      const double to_floor = 0.08 + pad.x() - p.x() - body.capsule.radius - floor;
      const double approach = s.s * guard.model().chain.point_velocity(f, body.capsule.link, p, dq).x() / dt;
      worst = std::max(worst, approach - guard.params().approach_gain * to_floor);
    }
  }
  check(worst <= 1e-9, "a part approaches faster than k times its distance to the floor");
  check(worst >= -1e-9, "no part moves at its limit");
  // The scaled step is s times the step: the direction stays.
  const VectorXd scaled = s.s * dq;
  check(std::abs(scaled.normalized().dot(dq.normalized()) - 1.0) < 1e-12, "the scale turns the step");
}

// A pad with a 10 mm allowance below the desk top: v_touch at the top, half of it halfway down, 0 at the floor.
void speed_scale_obeys_a3b_for_a_pad_at_the_desk() {
  const VectorXd low = fr3::low();
  const Vector3d pad = pad_at(bare_guard(), low);
  const double b = 0.010, dt = 0.001;
  shared_kinematics::ArmModel only_pad = fr3::model();
  only_pad.bodies = {fr3::bodies().back()};
  const VectorXd dq = move_pad(bare_guard(), low, Vector3d(0, 0, -0.2 * dt), VectorXd::Zero(7)) - low;
  const double down = -bare_guard().model().chain.point_velocity(bare_guard().model().chain.frames(low),
                                                                  fr3::kFlange, pad, dq).z() / dt;
  for (const auto& [height, expected] : std::vector<std::pair<double, double>>{
           {0.0, 0.05}, {-0.005, 0.025}, {-0.010, 0.0}, {-0.012, 0.0}, {0.05, 0.05 + 2.0 * 0.05}}) {
    const Guard guard(only_pad, {plane("desk", Vector3d(0, 0, 1), pad - Vector3d(0, 0, height), b)});
    const auto s = guard.speed_scale(low, dq, dt);
    const double speed = s.s * down;
    check(std::abs(speed - std::min(expected, down)) < 1e-9,
          "at " + str(height) + " m the pad moves down at " + str(speed) + " m/s, not " + str(expected));
    check(guard.speed_scale(low, -dq, dt).s == 1.0, "the desk slows a pad that moves up");
  }
}

void speed_scale_slows_a_part_that_approaches_another_body() {
  const VectorXd low = fr3::low();
  const Vector3d pad = pad_at(bare_guard(), low);
  const OtherBody other = {{pad + Vector3d(0.08, -0.1, 0), pad + Vector3d(0.08, 0.1, 0), 0.0}};
  const double dt = 0.001;
  const VectorXd dq = move_pad(bare_guard(), low, Vector3d(0.5 * dt, 0, 0), VectorXd::Zero(7)) - low;
  const auto s = bare_guard().speed_scale(low, dq, dt, {other});
  check(s.slowest == std::optional<std::string>("other:0") && s.s < 0.5, "the other body does not slow the step");
  check(bare_guard().speed_scale(low, -dq, dt, {other}).s == 1.0, "the other body slows a step away from it");
}

// One link that turns about z at the origin and holds one capsule from a to b.
Guard one_joint_guard(const Vector3d& a, const Vector3d& b, double radius) {
  shared_kinematics::ArmModel m;
  const Vector3d z = Vector3d::UnitZ();
  m.chain = shared_kinematics::Chain({{"base", -1, Eigen::Isometry3d::Identity(), -1, z},
                                      {"arm", 0, Eigen::Isometry3d::Identity(), 0, z}});
  m.bodies = {{{"arm", 1, a, b, radius}, shared_kinematics::Role::kRigid, true}};
  return Guard(m);
}

void short_crossing_segments_are_not_parallel() {
  const auto [p, o] = shared_kinematics::closest_points_of_segments(Vector3d(0, 0, 0), Vector3d(0.2, 0, 0),
                                                                    Vector3d(0.1, -2e-6, 0), Vector3d(0.1, 2e-6, 0));
  check((p - o).norm() < 1e-9, "the crossing segments are " + str((p - o).norm()) + " m apart");
}

void a_move_away_from_the_nearest_capsule_of_another_body_goes_through() {
  const Guard guard = one_joint_guard(Vector3d(0.3, 0, 0), Vector3d(0.3, 0, 0), 0.01);
  // Clearance 0.04 m to the first capsule and 0.01 m to the second; the move takes the second to 0.02 m.
  const OtherBody other = {{Vector3d(0.3, 0, 0.05), Vector3d(0.3, 0, 0.05), 0.0},
                           {Vector3d(0.3, -0.02, 0), Vector3d(0.3, -0.02, 0), 0.0}};
  const VectorXd from = VectorXd::Zero(1), to = VectorXd::Constant(1, 0.033);
  const ClipResult r = guard.clip(from, to, {other});
  check(r.fraction == 1.0 && r.others_bound.empty(), "the move away stops at " + str(r.fraction));
}

void speed_scale_holds_the_fastest_end_of_a_part_parallel_to_another_body() {
  const Guard guard = one_joint_guard(Vector3d(0, 0, 0), Vector3d(0.1, 0, 0), 0.01);
  const OtherBody other = {{Vector3d(0, 0.071, 0), Vector3d(0.1, 0.071, 0), 0.01}};
  const double dt = 0.001;
  // The far end moves toward the other body at 0.1 m/s; 1 mm to its floor allows 0.002 m/s.
  const auto s = guard.speed_scale(VectorXd::Zero(1), VectorXd::Constant(1, dt), dt, {other});
  check(std::abs(s.s - 0.02) < 1e-6, "the step keeps scale " + str(s.s));
}

}  // namespace

int main() {
  const std::vector<std::pair<const char*, std::function<void()>>> tests = {
      {"the_chain_frames_match_the_fr3_dh_frames_within_0_5_mm", the_chain_frames_match_the_fr3_dh_frames_within_0_5_mm},
      {"the_point_velocity_is_the_derivative_of_the_point_position",
       the_point_velocity_is_the_derivative_of_the_point_position},
      {"the_self_clearances_and_rate_bounds_equal_those_of_32", the_self_clearances_and_rate_bounds_equal_those_of_32},
      {"with_no_box_clip_equals_the_walk_of_32_on_synthetic_moves",
       with_no_box_clip_equals_the_walk_of_32_on_synthetic_moves},
      {"with_no_box_clip_equals_the_walk_of_32_on_the_recorded_replay",
       with_no_box_clip_equals_the_walk_of_32_on_the_recorded_replay},
      {"with_no_box_and_no_other_body_speed_scale_is_one_and_clip_keeps_the_self_pairs_only",
       with_no_box_and_no_other_body_speed_scale_is_one_and_clip_keeps_the_self_pairs_only},
      {"a_target_past_a_wall_stops_within_1_mm_of_the_floor_in_true_fk",
       a_target_past_a_wall_stops_within_1_mm_of_the_floor_in_true_fk},
      {"a_point_past_a_floor_can_move_back_in_only", a_point_past_a_floor_can_move_back_in_only},
      {"the_slide_keeps_the_motion_along_a_side", the_slide_keeps_the_motion_along_a_side},
      {"a_corner_of_two_sides_binds_both", a_corner_of_two_sides_binds_both},
      {"a_move_that_crosses_the_other_arms_committed_move_stops_before_it",
       a_move_that_crosses_the_other_arms_committed_move_stops_before_it},
      {"speed_scale_is_one_far_from_all_sides", speed_scale_is_one_far_from_all_sides},
      {"speed_scale_keeps_the_step_direction_and_obeys_a3", speed_scale_keeps_the_step_direction_and_obeys_a3},
      {"speed_scale_obeys_a3b_for_a_pad_at_the_desk", speed_scale_obeys_a3b_for_a_pad_at_the_desk},
      {"short_crossing_segments_are_not_parallel", short_crossing_segments_are_not_parallel},
      {"a_move_away_from_the_nearest_capsule_of_another_body_goes_through",
       a_move_away_from_the_nearest_capsule_of_another_body_goes_through},
      {"speed_scale_holds_the_fastest_end_of_a_part_parallel_to_another_body",
       speed_scale_holds_the_fastest_end_of_a_part_parallel_to_another_body},
      {"speed_scale_slows_a_part_that_approaches_another_body",
       speed_scale_slows_a_part_that_approaches_another_body},
  };
  int failed = 0, skipped = 0;
  for (const auto& [name, test] : tests) {
    try {
      test();
      std::cout << "PASS " << name << "\n";
    } catch (const Skip&) {
      std::cout << "SKIP " << name << "\n";
      ++skipped;
    } catch (const Failure& f) {
      std::cout << "FAIL " << name << ": " << f.what << "\n";
      ++failed;
    } catch (const std::exception& e) {
      std::cout << "FAIL " << name << ": " << e.what() << "\n";
      ++failed;
    }
  }
  std::cout << tests.size() - failed - skipped << " passed, " << failed << " failed, " << skipped << " skipped\n";
  return failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
