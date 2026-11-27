#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace shared_kinematics {

// One link of a kinematic tree. A revolute joint turns the link about `axis` after `origin`.
struct Link {
  std::string name;
  int parent = -1;  // index of the parent link; -1 for the root
  Eigen::Isometry3d origin = Eigen::Isometry3d::Identity();  // the link frame in the parent frame at zero joints
  int joint = -1;  // index of the revolute joint that turns this link; -1 for a fixed link
  Eigen::Vector3d axis = Eigen::Vector3d::UnitZ();  // unit, in the link frame
};

// Base-frame transforms of all links, in link order.
using Frames = std::vector<Eigen::Isometry3d>;

// A tree of links with revolute joints. Link 0 is the root and its frame is the base frame.
// The links come in an order where each parent comes before its children.
class Chain {
 public:
  Chain() = default;

  explicit Chain(std::vector<Link> links) : links_(std::move(links)) {
    if (links_.empty()) throw std::invalid_argument("a chain needs at least one link");
    if (links_[0].parent != -1 || links_[0].joint != -1)
      throw std::invalid_argument("the root link has a parent or a joint");
    int joints = 0;
    for (const Link& link : links_) joints += link.joint >= 0 ? 1 : 0;
    joint_link_.assign(joints, -1);
    moving_.resize(links_.size());
    for (std::size_t i = 0; i < links_.size(); ++i) {
      Link& link = links_[i];
      if (i > 0 && (link.parent < 0 || link.parent >= static_cast<int>(i)))
        throw std::invalid_argument("link " + link.name + " does not come after its parent");
      if (!link.origin.matrix().allFinite()) throw std::invalid_argument("link " + link.name + " has no finite origin");
      if (link.joint >= 0) {
        if (link.joint >= joints || joint_link_[link.joint] >= 0)
          throw std::invalid_argument("the joints of the chain are not 0 .. n-1, each once");
        const double norm = link.axis.norm();
        if (!(norm > 1e-9)) throw std::invalid_argument("link " + link.name + " has no joint axis");
        link.axis /= norm;
        joint_link_[link.joint] = static_cast<int>(i);
      }
      if (i > 0) moving_[i] = moving_[link.parent];
      if (link.joint >= 0) moving_[i].push_back(link.joint);
    }
  }

  int joint_count() const { return static_cast<int>(joint_link_.size()); }
  int link_count() const { return static_cast<int>(links_.size()); }
  const Link& link(int i) const { return links_.at(i); }
  const std::vector<Link>& links() const { return links_; }

  // The index of the link named `name`, or -1.
  int find(const std::string& name) const {
    for (std::size_t i = 0; i < links_.size(); ++i)
      if (links_[i].name == name) return static_cast<int>(i);
    return -1;
  }

  // The joints that move `link`, from the root.
  const std::vector<int>& moving_joints(int link) const { return moving_.at(link); }

  // The link that joint `j` turns.
  int joint_link(int j) const { return joint_link_.at(j); }

  void check_joints(const Eigen::VectorXd& q) const {
    if (q.size() != joint_count())
      throw std::invalid_argument("expected " + std::to_string(joint_count()) + " joints, got " +
                                  std::to_string(q.size()));
    if (!q.allFinite()) throw std::invalid_argument("the joints are not finite");
  }

  void frames(const Eigen::VectorXd& q, Frames& f) const {
    f.resize(links_.size());
    for (std::size_t i = 0; i < links_.size(); ++i) {
      const Link& link = links_[i];
      f[i] = i == 0 ? link.origin : f[link.parent] * link.origin;
      if (link.joint >= 0) f[i] = f[i] * Eigen::AngleAxisd(q(link.joint), link.axis);
    }
  }

  Frames frames(const Eigen::VectorXd& q) const {
    Frames f;
    frames(q, f);
    return f;
  }

  // How fast a point fixed to `link` moves along `direction`, per radian of each joint. `p` is in the base frame.
  void point_speed_along(const Frames& f, int link, const Eigen::Vector3d& p, const Eigen::Vector3d& direction,
                         Eigen::VectorXd& g) const {
    g.setZero(joint_count());
    for (int j : moving_[link]) {
      const Eigen::Isometry3d& frame = f[joint_link_[j]];
      const Eigen::Vector3d axis = frame.linear() * links_[joint_link_[j]].axis;
      g(j) = direction.dot(axis.cross(p - frame.translation()));
    }
  }

  // The base-frame velocity of a point fixed to `link` when the joints move by `dq`.
  Eigen::Vector3d point_velocity(const Frames& f, int link, const Eigen::Vector3d& p, const Eigen::VectorXd& dq) const {
    Eigen::Vector3d v = Eigen::Vector3d::Zero();
    for (int j : moving_[link]) {
      const Eigen::Isometry3d& frame = f[joint_link_[j]];
      const Eigen::Vector3d axis = frame.linear() * links_[joint_link_[j]].axis;
      v += dq(j) * axis.cross(p - frame.translation());
    }
    return v;
  }

  // For each link, (w, c): a point p fixed to the link moves by w x p + c when the joints move by `dq`.
  void link_motions(const Frames& f, const Eigen::VectorXd& dq,
                    std::vector<std::pair<Eigen::Vector3d, Eigen::Vector3d>>& out) const {
    out.resize(links_.size());
    for (std::size_t i = 0; i < links_.size(); ++i) {
      out[i] = i == 0 ? std::make_pair(Eigen::Vector3d::Zero().eval(), Eigen::Vector3d::Zero().eval())
                      : out[links_[i].parent];
      const int j = links_[i].joint;
      if (j < 0) continue;
      const Eigen::Vector3d turn = dq(j) * (f[i].linear() * links_[i].axis);
      out[i].first += turn;
      out[i].second -= turn.cross(f[i].translation());
    }
  }

  // Raises bound(j), for each j in `joints`, to the farthest that a point within `reach` of `link` can be from j.
  void reach_bounds(int link, double reach, const std::vector<int>& joints, Eigen::VectorXd& bound) const {
    for (int l = link; l >= 0; l = links_[l].parent) {
      const int j = links_[l].joint;
      if (j >= 0)
        for (int k : joints)
          if (k == j) bound(j) = std::max(bound(j), reach);
      reach += links_[l].origin.translation().norm();
    }
  }

 private:
  std::vector<Link> links_;
  std::vector<int> joint_link_;
  std::vector<std::vector<int>> moving_;
};

}  // namespace shared_kinematics
