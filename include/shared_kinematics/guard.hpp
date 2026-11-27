#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/QR>

#include "chain.hpp"
#include "geometry.hpp"
#include "model.hpp"

namespace shared_kinematics {

// A `surface` side is a real desk or wall; a `virtual` side is a line that an operator sets.
enum class SideKind { kSurface, kVirtual };

// One side of the box in the rig frame. A point x is inside when n . x + d >= 0.
// An allowance is how far past the side a part may go; a negative allowance keeps the part inside by a margin.
struct Plane {
  std::string name;
  SideKind kind = SideKind::kSurface;
  Eigen::Vector3d n = Eigen::Vector3d::UnitZ();
  double d = 0.0;
  double pad_allowance = 0.0;
  double rigid_allowance = 0.0;
};

struct GuardParams {
  // How far below its floor a check may fall between two checked points of a move, m.
  double path_tolerance = 0.001;
  int least_samples = 16;
  int bisections = 8;
  // Near a floor, a part moves toward it at most this gain times its distance to the floor, 1/s.
  double approach_gain = 2.0;
  // A pad meets the top of a side that it may pass at most at this speed, m/s.
  double touch_speed = 0.05;
  // The least clearance to another body, m.
  double other_clearance = 0.05;
  // The most that a part moves between two capsule sets of a swept move, m.
  double swept_spacing = 0.01;
  int max_swept_samples = 64;
};

// A capsule in the rig frame, for example a part of the other arm.
struct WorldCapsule {
  Eigen::Vector3d a = Eigen::Vector3d::Zero();
  Eigen::Vector3d b = Eigen::Vector3d::Zero();
  double radius = 0.0;
};

// The capsules of one other body, or the capsules that one committed move of it sweeps.
using OtherBody = std::vector<WorldCapsule>;

struct ClipResult {
  Eigen::VectorXd q;                     // the committed target
  double fraction = 1.0;                 // the part of the straight move that the first walk keeps
  std::vector<std::string> bound_sides;  // the sides that stopped the move
  bool self_bound = false;               // a self pair stopped the move
  std::vector<int> others_bound;         // the other bodies that stopped the move
  bool slid = false;                     // the target slid along the binding checks
};

struct ScaleResult {
  double s = 1.0;
  std::optional<std::string> slowest;  // the side, or "other:<index>", that set s; none when s is 1
};

struct PlaneDistance {
  std::string side;
  std::string body;
  double distance;  // m past the side is negative
  double floor;     // the least distance that the box allows
  Eigen::VectorXd gradient;
};

struct PairDistance {
  std::string pair;
  double clearance;
  Eigen::VectorXd gradient;
};

struct OtherDistance {
  int other;
  std::string body;
  double clearance;
  Eigen::VectorXd gradient;
};

struct Distances {
  std::vector<PlaneDistance> planes;
  std::vector<PairDistance> pairs;
  std::vector<OtherDistance> others;
};

// The box and clearance guard of one arm.
class Guard {
 public:
  explicit Guard(ArmModel model, std::vector<Plane> planes = {}, GuardParams params = {})
      : model_(std::move(model)), planes_(std::move(planes)), params_(params) {
    model_.validate();
    if (!(params_.path_tolerance > 0) || params_.least_samples < 1 || params_.bisections < 0 ||
        !(params_.approach_gain >= 0) || !(params_.touch_speed >= 0) || !std::isfinite(params_.other_clearance) ||
        !(params_.swept_spacing > 0) || params_.max_swept_samples < 1)
      throw std::invalid_argument("a guard parameter is out of range");
    const Eigen::Isometry3d& base_in_rig = model_.base_pose;
    for (Plane& plane : planes_) {
      const double norm = plane.n.norm();
      if (!(norm > 1e-9) || !plane.n.allFinite() || !std::isfinite(plane.d) ||
          !std::isfinite(plane.pad_allowance) || !std::isfinite(plane.rigid_allowance))
        throw std::invalid_argument("side " + plane.name + " is not a plane");
      plane.n /= norm;
      plane.d /= norm;
      // n . (R x + t) + d = (R^T n) . x + (n . t + d)
      base_planes_.push_back(
          {base_in_rig.linear().transpose() * plane.n, plane.n.dot(base_in_rig.translation()) + plane.d});
    }
    const int nj = model_.chain.joint_count();
    for (std::size_t i = 0; i < model_.bodies.size(); ++i)
      if (model_.bodies[i].in_planes) plane_bodies_.push_back(static_cast<int>(i));
    body_reach_.assign(model_.bodies.size(), Eigen::VectorXd::Zero(nj));
    plane_rate_ = Eigen::VectorXd::Zero(nj);
    body_rate_ = Eigen::VectorXd::Zero(nj);
    for (std::size_t i = 0; i < model_.bodies.size(); ++i) {
      const Capsule& c = model_.bodies[i].capsule;
      model_.chain.reach_bounds(c.link, std::max(c.a.norm(), c.b.norm()), model_.chain.moving_joints(c.link),
                                body_reach_[i]);
      body_rate_ = body_rate_.cwiseMax(body_reach_[i]);
      if (model_.bodies[i].in_planes) plane_rate_ = plane_rate_.cwiseMax(body_reach_[i]);
    }
    self_rate_ = Eigen::VectorXd::Zero(nj);
    for (const SelfPair& pair : model_.self.pairs)
      for (int i : pair.a)
        for (int j : pair.b) {
          add_self_rate(model_.self.capsules[j], model_.self.capsules[i]);
          add_self_rate(model_.self.capsules[i], model_.self.capsules[j]);
        }
  }

  const ArmModel& model() const { return model_; }
  const std::vector<Plane>& planes() const { return planes_; }
  const GuardParams& params() const { return params_; }

  // For each joint, the most that a self-pair clearance changes per radian of that joint.
  const Eigen::VectorXd& self_rate_bounds() const { return self_rate_; }
  // For each joint, the most that a plane distance changes per radian of that joint.
  const Eigen::VectorXd& plane_rate_bounds() const { return plane_rate_; }

  // The point of the straight joint move from q_from to q_to farthest along it whose path keeps each check at its
  // floor. The floor of a check is its limit, or its value at q_from when that is less. When a side or another
  // body stops the move, one slide drops the rest of the move along the binding checks and walks again.
  ClipResult clip(const Eigen::VectorXd& q_from, const Eigen::VectorXd& q_to,
                  const std::vector<OtherBody>& others = {}) const {
    model_.chain.check_joints(q_from);
    model_.chain.check_joints(q_to);
    Context ctx = context(others);
    ClipResult result;
    result.q = q_to;
    if (check_count(ctx) == 0) return result;
    const std::vector<double> floors = floors_at(ctx, q_from);
    const Walk first = walk(ctx, floors, q_from, q_to);
    if (!first.stopped) return result;
    const Eigen::VectorXd stop = q_from + first.safe * (q_to - q_from);
    result.q = stop;
    result.fraction = first.safe;
    const std::vector<int> binding = failing(ctx, floors, q_from + first.blocked * (q_to - q_from));
    note_binding(ctx, binding, result);
    const bool slide =
        std::any_of(binding.begin(), binding.end(), [&](int k) { return locate(ctx, k).kind != kSelf; });
    if (!slide) return result;
    pose(ctx, stop);
    Eigen::MatrixXd g(binding.size(), q_from.size());
    for (std::size_t r = 0; r < binding.size(); ++r) g.row(r) = evaluate(ctx, binding[r], true).gradient.transpose();
    const Eigen::VectorXd rest = project_onto_cone(q_to - stop, g);
    if (rest.norm() < 1e-12) return result;
    // The fast loop plays the straight move from q_from, so the walk checks that move to the slid target.
    const Eigen::VectorXd end = stop + rest;
    const Walk second = walk(ctx, floors, q_from, end);
    const Eigen::VectorXd slid = second.stopped ? Eigen::VectorXd(q_from + second.safe * (end - q_from)) : end;
    if ((slid - q_to).norm() >= (stop - q_to).norm()) return result;
    result.q = slid;
    result.slid = true;
    if (second.stopped) note_binding(ctx, failing(ctx, floors, q_from + second.blocked * (end - q_from)), result);
    return result;
  }

  // One factor in [0, 1] on the joint step dq_step of a tick of dt seconds. Near a floor, each part moves toward
  // it at most approach_gain times its distance to the floor. A pad that may pass a side meets the side at most at
  // touch_speed, and slows linearly to 0 at its floor.
  ScaleResult speed_scale(const Eigen::VectorXd& q, const Eigen::VectorXd& dq_step, double dt,
                          const std::vector<OtherBody>& others = {}) const {
    model_.chain.check_joints(q);
    model_.chain.check_joints(dq_step);
    if (!(dt > 0) || !std::isfinite(dt)) throw std::invalid_argument("dt is not a positive time");
    ScaleResult result;
    if (planes_.empty() && others.empty()) return result;
    Context ctx = context(others);
    pose(ctx, q, false);
    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>> motion;
    model_.chain.link_motions(ctx.f, dq_step, motion);
    const auto velocity = [&](int link, const Eigen::Vector3d& p) {
      return Eigen::Vector3d((motion[link].first.cross(p) + motion[link].second) / dt);
    };
    const auto limit = [&](double approach, double allowed, const std::string& name) {
      if (approach <= 0 || allowed >= result.s * approach) return;
      result.s = std::max(0.0, allowed / approach);
      result.slowest = name;
    };
    for (std::size_t p = 0; p < planes_.size(); ++p) {
      const auto& [n, d] = base_planes_[p];
      for (int i : plane_bodies_) {
        const Body& body = model_.bodies[i];
        const double allowance = allowance_of(planes_[p], body.role);
        for (const Eigen::Vector3d* end : {&ctx.body_seg[i].first, &ctx.body_seg[i].second}) {
          const double to_floor = n.dot(*end) + d - body.capsule.radius + allowance;
          const double approach = -n.dot(velocity(body.capsule.link, *end));
          limit(approach, allowed_speed(body.role, allowance, to_floor), planes_[p].name);
        }
      }
    }
    for (std::size_t k = 0; k < ctx.others.size(); ++k) {
      const std::string name = "other:" + std::to_string(k);
      for (std::size_t i = 0; i < model_.bodies.size(); ++i) {
        const Body& body = model_.bodies[i];
        const auto& [a, b] = ctx.body_seg[i];
        const Eigen::Vector3d center = 0.5 * (a + b);
        const double bound = 0.5 * (b - a).norm() + body.capsule.radius;
        const double fastest = body_reach_[i].dot(dq_step.cwiseAbs()) / dt;
        for (const BaseCapsule& c : ctx.others[k]) {
          const double least = (center - c.center).norm() - bound - c.bound - params_.other_clearance;
          if (least > 0 && params_.approach_gain * least >= result.s * fastest) continue;
          const auto [p, o] = closest_points_of_segments(a, b, c.a, c.b);
          const double distance = (p - o).norm();
          const double to_floor = distance - body.capsule.radius - c.radius - params_.other_clearance;
          if (distance < 1e-12) {
            limit(fastest, 0.0, name);
            continue;
          }
          const Eigen::Vector3d away = (p - o) / distance;
          const double approach = -away.dot(velocity(body.capsule.link, p));
          limit(approach, params_.approach_gain * std::max(0.0, to_floor), name);
        }
      }
    }
    return result;
  }

  // Each plane distance of each bounded part, each self-pair clearance, and each clearance to another body, with
  // its gradient in the joints.
  Distances distances(const Eigen::VectorXd& q, const std::vector<OtherBody>& others = {}) const {
    model_.chain.check_joints(q);
    Context ctx = context(others);
    pose(ctx, q);
    Distances out;
    for (int k = 0; k < check_count(ctx); ++k) {
      const Value v = evaluate(ctx, k, true);
      const Location at = locate(ctx, k);
      if (at.kind == kSelf) {
        out.pairs.push_back({model_.self.pairs[at.index].name, v.value, v.gradient});
      } else if (at.kind == kPlane) {
        const Body& b = model_.bodies[at.body];
        out.planes.push_back(
            {planes_[at.index].name, b.capsule.name, v.value, -allowance_of(planes_[at.index], b.role), v.gradient});
      } else {
        out.others.push_back({at.index, model_.bodies[at.body].capsule.name, v.value, v.gradient});
      }
    }
    return out;
  }

  // The capsules of every body at q, in the rig frame.
  OtherBody capsules_at(const Eigen::VectorXd& q) const {
    model_.chain.check_joints(q);
    return sweep(q, q, 0);
  }

  // Capsules in the rig frame that hold every body over the straight joint move from q_from to q_to.
  OtherBody swept_capsules(const Eigen::VectorXd& q_from, const Eigen::VectorXd& q_to) const {
    model_.chain.check_joints(q_from);
    model_.chain.check_joints(q_to);
    double longest = 0.0;
    for (const Eigen::VectorXd& reach : body_reach_)
      longest = std::max(longest, reach.dot((q_to - q_from).cwiseAbs()));
    const int steps =
        std::clamp(static_cast<int>(std::ceil(longest / params_.swept_spacing)), 1, params_.max_swept_samples);
    return sweep(q_from, q_to, steps);
  }

 private:
  enum Kind { kSelf, kPlane, kOther };

  struct BaseCapsule {
    Eigen::Vector3d a, b;
    double radius;
    Eigen::Vector3d center;
    double bound;  // the farthest point of the capsule from its center
  };

  using Segment = std::pair<Eigen::Vector3d, Eigen::Vector3d>;

  struct Context {
    std::vector<std::vector<BaseCapsule>> others;
    Frames f;
    std::vector<Segment> self_seg;
    std::vector<Segment> body_seg;
  };

  // For each other body and each bounded body, the capsules of the other body that one walk can bring below the
  // floor. Empty: every capsule.
  using Near = std::vector<std::vector<std::vector<int>>>;

  struct Value {
    double value = std::numeric_limits<double>::infinity();
    Eigen::VectorXd gradient;
  };

  struct Walk {
    double safe = 1.0;
    double blocked = 1.0;
    bool stopped = false;
  };

  struct Location {
    Kind kind;
    int index;  // the self pair, the plane, or the other body
    int body;   // the bounded body; -1 for a self pair
  };

  void add_self_rate(const Capsule& moving, const Capsule& other) {
    std::vector<int> joints;
    const auto& theirs = model_.chain.moving_joints(other.link);
    for (int j : model_.chain.moving_joints(moving.link))
      if (std::find(theirs.begin(), theirs.end(), j) == theirs.end()) joints.push_back(j);
    model_.chain.reach_bounds(moving.link, std::max(moving.a.norm(), moving.b.norm()), joints, self_rate_);
  }

  static double allowance_of(const Plane& plane, Role role) {
    return role == Role::kPad ? plane.pad_allowance : plane.rigid_allowance;
  }

  double allowed_speed(Role role, double allowance, double to_floor) const {
    if (to_floor <= 0) return 0.0;
    if (role == Role::kPad && allowance > 0) {
      if (to_floor < allowance) return params_.touch_speed * to_floor / allowance;
      return params_.touch_speed + params_.approach_gain * (to_floor - allowance);
    }
    return params_.approach_gain * to_floor;
  }

  Context context(const std::vector<OtherBody>& others) const {
    Context ctx;
    const Eigen::Isometry3d rig_to_base = model_.base_pose.inverse();
    for (const OtherBody& other : others) {
      std::vector<BaseCapsule> capsules;
      capsules.reserve(other.size());
      for (const WorldCapsule& w : other) {
        if (!w.a.allFinite() || !w.b.allFinite() || !std::isfinite(w.radius) || w.radius < 0)
          throw std::invalid_argument("an other body has a capsule with no finite shape");
        const Eigen::Vector3d a = rig_to_base * w.a, b = rig_to_base * w.b;
        capsules.push_back({a, b, w.radius, 0.5 * (a + b), 0.5 * (b - a).norm() + w.radius});
      }
      ctx.others.push_back(std::move(capsules));
    }
    return ctx;
  }

  int check_count(const Context& ctx) const {
    return static_cast<int>(model_.self.pairs.size() + planes_.size() * plane_bodies_.size() +
                            ctx.others.size() * model_.bodies.size());
  }

  Location locate(const Context& ctx, int k) const {
    const int pairs = static_cast<int>(model_.self.pairs.size());
    if (k < pairs) return {kSelf, k, -1};
    k -= pairs;
    const int per_plane = static_cast<int>(plane_bodies_.size());
    const int in_planes = static_cast<int>(planes_.size()) * per_plane;
    if (k < in_planes) return {kPlane, k / per_plane, plane_bodies_[k % per_plane]};
    k -= in_planes;
    const int per_other = static_cast<int>(model_.bodies.size());
    if (k >= static_cast<int>(ctx.others.size()) * per_other) throw std::out_of_range("no such check");
    return {kOther, k / per_other, k % per_other};
  }

  void pose(Context& ctx, const Eigen::VectorXd& q, bool self = true) const {
    model_.chain.frames(q, ctx.f);
    ctx.self_seg.resize(self ? model_.self.capsules.size() : 0);
    for (std::size_t i = 0; i < ctx.self_seg.size(); ++i) {
      const Capsule& c = model_.self.capsules[i];
      ctx.self_seg[i] = {ctx.f[c.link] * c.a, ctx.f[c.link] * c.b};
    }
    ctx.body_seg.resize(model_.bodies.size());
    for (std::size_t i = 0; i < model_.bodies.size(); ++i) {
      const Capsule& c = model_.bodies[i].capsule;
      ctx.body_seg[i] = {ctx.f[c.link] * c.a, ctx.f[c.link] * c.b};
    }
  }

  // The value of check k at the pose in ctx. Without the gradient, an other-body check may stop at the first
  // capsule below `below`. An other-body check reads only the capsules in `near`.
  Value evaluate(const Context& ctx, int k, bool with_gradient,
                 double below = std::numeric_limits<double>::infinity(), const Near& near = {}) const {
    const Location at = locate(ctx, k);
    Value v;
    if (with_gradient) v.gradient = Eigen::VectorXd::Zero(model_.chain.joint_count());
    if (at.kind == kSelf) {
      Eigen::VectorXd ga, gb;
      const SelfPair& pair = model_.self.pairs[at.index];
      for (int i : pair.a)
        for (int j : pair.b) {
          const Capsule &ca = model_.self.capsules[i], &cb = model_.self.capsules[j];
          const auto [pa, pb] = closest_points_of_segments(ctx.self_seg[i].first, ctx.self_seg[i].second,
                                                           ctx.self_seg[j].first, ctx.self_seg[j].second);
          const double distance = (pa - pb).norm();
          const double clearance = distance - ca.radius - cb.radius;
          if (!(clearance < v.value)) continue;
          v.value = clearance;
          if (!with_gradient) continue;
          // Crossed axes give no direction to push apart.
          const Eigen::Vector3d n = distance > 1e-9 ? Eigen::Vector3d((pa - pb) / distance) : Eigen::Vector3d::Zero();
          model_.chain.point_speed_along(ctx.f, ca.link, pa, n, ga);
          model_.chain.point_speed_along(ctx.f, cb.link, pb, n, gb);
          v.gradient = ga - gb;
        }
      return v;
    }
    const Body& body = model_.bodies[at.body];
    const Segment& seg = ctx.body_seg[at.body];
    if (at.kind == kPlane) {
      const auto& [n, d] = base_planes_[at.index];
      const double da = n.dot(seg.first), db = n.dot(seg.second);
      v.value = std::min(da, db) + d - body.capsule.radius;
      if (with_gradient)
        model_.chain.point_speed_along(ctx.f, body.capsule.link, da <= db ? seg.first : seg.second, n, v.gradient);
      return v;
    }
    const Eigen::Vector3d center = 0.5 * (seg.first + seg.second);
    const double bound = 0.5 * (seg.second - seg.first).norm() + body.capsule.radius;
    const std::vector<BaseCapsule>& capsules = ctx.others[at.index];
    const std::vector<int>* kept = near.empty() ? nullptr : &near[at.index][at.body];
    const std::size_t count = kept ? kept->size() : capsules.size();
    for (std::size_t n = 0; n < count; ++n) {
      const BaseCapsule& c = capsules[kept ? (*kept)[n] : n];
      if ((center - c.center).norm() - bound - c.bound >= std::min(v.value, below)) continue;
      const auto [p, o] = closest_points_of_segments(seg.first, seg.second, c.a, c.b);
      const double distance = (p - o).norm();
      const double clearance = distance - body.capsule.radius - c.radius;
      if (!(clearance < v.value)) continue;
      v.value = clearance;
      if (with_gradient) {
        const Eigen::Vector3d n = distance > 1e-9 ? Eigen::Vector3d((p - o) / distance) : Eigen::Vector3d::Zero();
        model_.chain.point_speed_along(ctx.f, body.capsule.link, p, n, v.gradient);
      } else if (v.value < below) {
        return v;
      }
    }
    return v;
  }

  double limit_of(const Context& ctx, int k) const {
    const Location at = locate(ctx, k);
    if (at.kind == kSelf) return model_.self.clearance;
    if (at.kind == kPlane) return -allowance_of(planes_[at.index], model_.bodies[at.body].role);
    return params_.other_clearance;
  }

  std::vector<double> floors_at(Context& ctx, const Eigen::VectorXd& q) const {
    pose(ctx, q);
    std::vector<double> floors(check_count(ctx));
    for (int k = 0; k < check_count(ctx); ++k) {
      const double limit = limit_of(ctx, k);
      floors[k] = std::min(limit, evaluate(ctx, k, false, limit).value);
    }
    return floors;
  }

  bool clear(Context& ctx, const std::vector<double>& floors, const Eigen::VectorXd& q, const Near& near) const {
    pose(ctx, q);
    for (int k = 0; k < check_count(ctx); ++k)
      if (evaluate(ctx, k, false, floors[k], near).value < floors[k]) return false;
    return true;
  }

  std::vector<int> failing(Context& ctx, const std::vector<double>& floors, const Eigen::VectorXd& q) const {
    pose(ctx, q);
    std::vector<int> out;
    for (int k = 0; k < check_count(ctx); ++k)
      if (evaluate(ctx, k, false, floors[k]).value < floors[k]) out.push_back(k);
    return out;
  }

  void note_binding(const Context& ctx, const std::vector<int>& binding, ClipResult& result) const {
    for (int k : binding) {
      const Location at = locate(ctx, k);
      if (at.kind == kSelf) {
        result.self_bound = true;
      } else if (at.kind == kPlane) {
        const std::string& name = planes_[at.index].name;
        if (std::find(result.bound_sides.begin(), result.bound_sides.end(), name) == result.bound_sides.end())
          result.bound_sides.push_back(name);
      } else if (std::find(result.others_bound.begin(), result.others_bound.end(), at.index) ==
                 result.others_bound.end()) {
        result.others_bound.push_back(at.index);
      }
    }
  }

  // For each bounded body, the other capsules that it can come below its floor to on a move of `moved` from `from`.
  Near near_on_move(Context& ctx, const std::vector<double>& floors, const Eigen::VectorXd& from,
                    const Eigen::VectorXd& moved) const {
    Near near;
    if (ctx.others.empty()) return near;
    pose(ctx, from);
    const int first = static_cast<int>(model_.self.pairs.size() + planes_.size() * plane_bodies_.size());
    const int per_other = static_cast<int>(model_.bodies.size());
    near.resize(ctx.others.size());
    for (std::size_t k = 0; k < ctx.others.size(); ++k) {
      near[k].resize(model_.bodies.size());
      for (std::size_t i = 0; i < model_.bodies.size(); ++i) {
        const Segment& seg = ctx.body_seg[i];
        const Eigen::Vector3d center = 0.5 * (seg.first + seg.second);
        const double reach = 0.5 * (seg.second - seg.first).norm() + model_.bodies[i].capsule.radius +
                             body_reach_[i].dot(moved);
        const double floor = floors[first + static_cast<int>(k) * per_other + static_cast<int>(i)];
        for (std::size_t c = 0; c < ctx.others[k].size(); ++c)
          if ((center - ctx.others[k][c].center).norm() - reach - ctx.others[k][c].bound < floor)
            near[k][i].push_back(static_cast<int>(c));
      }
    }
    return near;
  }

  // The walk of positronic-franka#32: enough samples that a check falls at most path_tolerance below its floor
  // between two of them, at least least_samples, then bisections at the first blocked sample.
  Walk walk(Context& ctx, const std::vector<double>& floors, const Eigen::VectorXd& from,
            const Eigen::VectorXd& to) const {
    const Eigen::VectorXd moved = (to - from).cwiseAbs();
    double change_over_move = self_rate_.dot(moved);
    if (!planes_.empty()) change_over_move = std::max(change_over_move, plane_rate_.dot(moved));
    if (!ctx.others.empty()) change_over_move = std::max(change_over_move, body_rate_.dot(moved));
    const int samples = std::max(params_.least_samples,
                                 static_cast<int>(std::ceil(change_over_move / (2 * params_.path_tolerance))));
    const Near near = near_on_move(ctx, floors, from, moved);
    double safe = 0.0;
    for (int i = 1; i <= samples; ++i) {
      const double s = static_cast<double>(i) / samples;
      if (clear(ctx, floors, from + s * (to - from), near)) {
        safe = s;
        continue;
      }
      double blocked = s;
      for (int b = 0; b < params_.bisections; ++b) {
        const double mid = 0.5 * (safe + blocked);
        (clear(ctx, floors, from + mid * (to - from), near) ? safe : blocked) = mid;
      }
      return {safe, blocked, true};
    }
    return {};
  }

  // The move nearest to `move` that lowers no check whose gradient is a row of g: min |x - move| subject to
  // g x >= 0, by an active set.
  static Eigen::VectorXd project_onto_cone(const Eigen::VectorXd& move, const Eigen::MatrixXd& g) {
    const int m = static_cast<int>(g.rows());
    std::vector<bool> active(m, false);
    for (int k = 0; k < m; ++k) active[k] = g.row(k).dot(move) < 0;
    for (int pass = 0; pass < 4 * m + 4; ++pass) {
      std::vector<int> rows;
      for (int k = 0; k < m; ++k)
        if (active[k]) rows.push_back(k);
      Eigen::VectorXd x = move, mu;
      if (!rows.empty()) {
        Eigen::MatrixXd ga(rows.size(), move.size());
        for (std::size_t r = 0; r < rows.size(); ++r) ga.row(r) = g.row(rows[r]);
        // x = move + ga^T mu, with ga x = 0.
        mu = (ga * ga.transpose()).completeOrthogonalDecomposition().solve(-ga * move);
        x = move + ga.transpose() * mu;
      }
      int drop = -1;
      for (std::size_t r = 0; r < rows.size(); ++r)
        if (mu(r) < -1e-12 && (drop < 0 || mu(r) < mu(drop))) drop = static_cast<int>(r);
      if (drop >= 0) {
        active[rows[drop]] = false;
        continue;
      }
      int add = -1;
      double worst = -1e-12;
      for (int k = 0; k < m; ++k)
        if (!active[k] && g.row(k).dot(x) < worst) {
          worst = g.row(k).dot(x);
          add = k;
        }
      if (add < 0) return x;
      active[add] = true;
    }
    return Eigen::VectorXd::Zero(move.size());
  }

  OtherBody sweep(const Eigen::VectorXd& from, const Eigen::VectorXd& to, int steps) const {
    OtherBody out;
    out.reserve(model_.bodies.size() * (steps + 1));
    const Eigen::VectorXd step = (to - from).cwiseAbs() / std::max(steps, 1);
    Frames f;
    for (int k = 0; k <= steps; ++k) {
      const Eigen::VectorXd q =
          steps == 0 ? from : Eigen::VectorXd(from + (static_cast<double>(k) / steps) * (to - from));
      model_.chain.frames(q, f);
      for (std::size_t i = 0; i < model_.bodies.size(); ++i) {
        const Capsule& c = model_.bodies[i].capsule;
        // A point moves at most body_reach . step between two sets, so it is within half of that of one of them.
        const double grow = steps == 0 ? 0.0 : 0.5 * body_reach_[i].dot(step);
        out.push_back({model_.base_pose * (f[c.link] * c.a), model_.base_pose * (f[c.link] * c.b), c.radius + grow});
      }
    }
    return out;
  }

  ArmModel model_;
  std::vector<Plane> planes_;
  GuardParams params_;
  std::vector<std::pair<Eigen::Vector3d, double>> base_planes_;
  std::vector<int> plane_bodies_;
  std::vector<Eigen::VectorXd> body_reach_;
  Eigen::VectorXd self_rate_, plane_rate_, body_rate_;
};

}  // namespace shared_kinematics
