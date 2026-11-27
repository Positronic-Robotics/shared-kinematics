#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "shared_kinematics/shared_kinematics.hpp"

namespace py = pybind11;
namespace sk = shared_kinematics;

namespace {

Eigen::Isometry3d isometry(const Eigen::Matrix4d& m) {
  if (!m.allFinite() || (m.row(3) - Eigen::RowVector4d(0, 0, 0, 1)).cwiseAbs().maxCoeff() > 1e-9)
    throw std::invalid_argument("a pose is a finite 4x4 homogeneous transform");
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.matrix() = m;
  return t;
}

}  // namespace

PYBIND11_MODULE(_core, m) {
  m.doc() = "The C++ core of shared_kinematics: the arm model, clip, speed_scale and distances.";

  py::class_<sk::Link>(m, "Link")
      .def(py::init([](std::string name, int parent, const Eigen::Matrix4d& origin, int joint,
                       const Eigen::Vector3d& axis, double lower, double upper) {
             return sk::Link{std::move(name), parent, isometry(origin), joint, axis, lower, upper};
           }),
           py::arg("name"), py::arg("parent"), py::arg("origin") = Eigen::Matrix4d::Identity().eval(),
           py::arg("joint") = -1, py::arg("axis") = Eigen::Vector3d::UnitZ().eval(),
           py::arg("lower") = -std::numeric_limits<double>::infinity(),
           py::arg("upper") = std::numeric_limits<double>::infinity())
      .def_readonly("name", &sk::Link::name)
      .def_readonly("parent", &sk::Link::parent)
      .def_property_readonly("origin", [](const sk::Link& l) { return Eigen::Matrix4d(l.origin.matrix()); })
      .def_readonly("joint", &sk::Link::joint)
      .def_readonly("axis", &sk::Link::axis)
      .def_readonly("lower", &sk::Link::lower)
      .def_readonly("upper", &sk::Link::upper);

  py::class_<sk::Chain>(m, "Chain")
      .def(py::init<std::vector<sk::Link>>(), py::arg("links"))
      .def_property_readonly("joint_count", &sk::Chain::joint_count)
      .def_property_readonly("links", &sk::Chain::links)
      .def("find", &sk::Chain::find, py::arg("name"))
      .def("moving_joints", &sk::Chain::moving_joints, py::arg("link"))
      .def(
          "frames",
          [](const sk::Chain& c, const Eigen::VectorXd& q) {
            c.check_joints(q);
            std::vector<Eigen::Matrix4d> out;
            for (const auto& f : c.frames(q)) out.push_back(f.matrix());
            return out;
          },
          py::arg("q"), "Base-frame 4x4 transforms of all links at q, in link order.");

  py::class_<sk::Capsule>(m, "Capsule")
      .def(py::init([](std::string name, int link, const Eigen::Vector3d& a, const Eigen::Vector3d& b, double radius) {
             return sk::Capsule{std::move(name), link, a, b, radius};
           }),
           py::arg("name"), py::arg("link"), py::arg("a"), py::arg("b"), py::arg("radius"))
      .def_readonly("name", &sk::Capsule::name)
      .def_readonly("link", &sk::Capsule::link)
      .def_readonly("a", &sk::Capsule::a)
      .def_readonly("b", &sk::Capsule::b)
      .def_readonly("radius", &sk::Capsule::radius);

  py::enum_<sk::Role>(m, "Role").value("PAD", sk::Role::kPad).value("RIGID", sk::Role::kRigid);

  py::class_<sk::Body>(m, "Body")
      .def(py::init<sk::Capsule, sk::Role, bool>(), py::arg("capsule"), py::arg("role") = sk::Role::kRigid,
           py::arg("in_planes") = true)
      .def_readonly("capsule", &sk::Body::capsule)
      .def_readonly("role", &sk::Body::role)
      .def_readonly("in_planes", &sk::Body::in_planes);

  py::enum_<sk::ParallelTest>(m, "ParallelTest")
      .value("ANGLE", sk::ParallelTest::kAngle)
      .value("FRANKA_32", sk::ParallelTest::kFranka32);

  py::class_<sk::SelfPair>(m, "SelfPair")
      .def(py::init<std::string, std::vector<int>, std::vector<int>, sk::ParallelTest>(), py::arg("name"),
           py::arg("a"), py::arg("b"), py::arg("parallel") = sk::ParallelTest::kAngle)
      .def_readonly("name", &sk::SelfPair::name)
      .def_readonly("a", &sk::SelfPair::a)
      .def_readonly("b", &sk::SelfPair::b)
      .def_readonly("parallel", &sk::SelfPair::parallel);

  py::class_<sk::SelfCollision>(m, "SelfCollision")
      .def(py::init<std::vector<sk::Capsule>, std::vector<sk::SelfPair>, double>(), py::arg("capsules"),
           py::arg("pairs"), py::arg("clearance"))
      .def_readonly("capsules", &sk::SelfCollision::capsules)
      .def_readonly("pairs", &sk::SelfCollision::pairs)
      .def_readonly("clearance", &sk::SelfCollision::clearance);

  py::class_<sk::ArmModel>(m, "ArmModel")
      .def(py::init([](sk::Chain chain, std::vector<sk::Body> bodies, sk::SelfCollision self,
                       const Eigen::Matrix4d& base_pose) {
             sk::ArmModel model{std::move(chain), std::move(bodies), std::move(self), isometry(base_pose)};
             model.validate();
             return model;
           }),
           py::arg("chain"), py::arg("bodies"), py::arg("self_collision"),
           py::arg("base_pose") = Eigen::Matrix4d::Identity().eval())
      .def_readonly("chain", &sk::ArmModel::chain)
      .def_readonly("bodies", &sk::ArmModel::bodies)
      .def_readonly("self_collision", &sk::ArmModel::self)
      .def_property_readonly("base_pose", [](const sk::ArmModel& a) { return Eigen::Matrix4d(a.base_pose.matrix()); });

  py::enum_<sk::SideKind>(m, "SideKind").value("SURFACE", sk::SideKind::kSurface).value("VIRTUAL", sk::SideKind::kVirtual);

  py::class_<sk::Plane>(m, "Plane")
      .def(py::init<std::string, sk::SideKind, Eigen::Vector3d, double, double, double>(), py::arg("name"),
           py::arg("kind"), py::arg("n"), py::arg("d"), py::arg("pad_allowance"), py::arg("rigid_allowance"))
      .def_readonly("name", &sk::Plane::name)
      .def_readonly("kind", &sk::Plane::kind)
      .def_readonly("n", &sk::Plane::n)
      .def_readonly("d", &sk::Plane::d)
      .def_readonly("pad_allowance", &sk::Plane::pad_allowance)
      .def_readonly("rigid_allowance", &sk::Plane::rigid_allowance);

  py::class_<sk::GuardParams>(m, "GuardParams")
      .def(py::init([](double path_tolerance, int least_samples, int bisections, double approach_gain,
                       double touch_speed, double other_clearance, double swept_spacing, int max_swept_samples) {
             return sk::GuardParams{path_tolerance, least_samples,   bisections,    approach_gain,
                                    touch_speed,    other_clearance, swept_spacing, max_swept_samples};
           }),
           py::kw_only(), py::arg("path_tolerance") = 0.001, py::arg("least_samples") = 16,
           py::arg("bisections") = 8, py::arg("approach_gain") = 2.0, py::arg("touch_speed") = 0.05,
           py::arg("other_clearance") = 0.05, py::arg("swept_spacing") = 0.01, py::arg("max_swept_samples") = 64)
      .def_readonly("path_tolerance", &sk::GuardParams::path_tolerance)
      .def_readonly("least_samples", &sk::GuardParams::least_samples)
      .def_readonly("bisections", &sk::GuardParams::bisections)
      .def_readonly("approach_gain", &sk::GuardParams::approach_gain)
      .def_readonly("touch_speed", &sk::GuardParams::touch_speed)
      .def_readonly("other_clearance", &sk::GuardParams::other_clearance)
      .def_readonly("swept_spacing", &sk::GuardParams::swept_spacing)
      .def_readonly("max_swept_samples", &sk::GuardParams::max_swept_samples);

  py::class_<sk::WorldCapsule>(m, "WorldCapsule")
      .def(py::init<Eigen::Vector3d, Eigen::Vector3d, double>(), py::arg("a"), py::arg("b"), py::arg("radius"))
      .def_readonly("a", &sk::WorldCapsule::a)
      .def_readonly("b", &sk::WorldCapsule::b)
      .def_readonly("radius", &sk::WorldCapsule::radius);

  py::class_<sk::ClipResult>(m, "ClipResult")
      .def_readonly("q", &sk::ClipResult::q)
      .def_readonly("fraction", &sk::ClipResult::fraction)
      .def_readonly("bound_sides", &sk::ClipResult::bound_sides)
      .def_readonly("self_bound", &sk::ClipResult::self_bound)
      .def_readonly("others_bound", &sk::ClipResult::others_bound)
      .def_readonly("slid", &sk::ClipResult::slid)
      .def_property_readonly("bound", [](const sk::ClipResult& r) {
        return r.self_bound || !r.bound_sides.empty() || !r.others_bound.empty();
      });

  py::class_<sk::ScaleResult>(m, "ScaleResult")
      .def_readonly("s", &sk::ScaleResult::s)
      .def_readonly("slowest", &sk::ScaleResult::slowest);

  py::class_<sk::PlaneDistance>(m, "PlaneDistance")
      .def_readonly("side", &sk::PlaneDistance::side)
      .def_readonly("body", &sk::PlaneDistance::body)
      .def_readonly("distance", &sk::PlaneDistance::distance)
      .def_readonly("floor", &sk::PlaneDistance::floor)
      .def_readonly("gradient", &sk::PlaneDistance::gradient);

  py::class_<sk::PairDistance>(m, "PairDistance")
      .def_readonly("pair", &sk::PairDistance::pair)
      .def_readonly("clearance", &sk::PairDistance::clearance)
      .def_readonly("gradient", &sk::PairDistance::gradient);

  py::class_<sk::OtherDistance>(m, "OtherDistance")
      .def_readonly("other", &sk::OtherDistance::other)
      .def_readonly("body", &sk::OtherDistance::body)
      .def_readonly("clearance", &sk::OtherDistance::clearance)
      .def_readonly("gradient", &sk::OtherDistance::gradient);

  py::class_<sk::Distances>(m, "Distances")
      .def_readonly("planes", &sk::Distances::planes)
      .def_readonly("pairs", &sk::Distances::pairs)
      .def_readonly("others", &sk::Distances::others);

  using Others = std::vector<sk::OtherBody>;
  py::class_<sk::Guard>(m, "Guard")
      .def(py::init<sk::ArmModel, std::vector<sk::Plane>, sk::GuardParams>(), py::arg("model"),
           py::arg("planes") = std::vector<sk::Plane>{}, py::arg("params") = sk::GuardParams{})
      .def_property_readonly("model", &sk::Guard::model)
      .def_property_readonly("planes", &sk::Guard::planes)
      .def_property_readonly("params", &sk::Guard::params)
      .def_property_readonly("self_rate_bounds", &sk::Guard::self_rate_bounds)
      .def_property_readonly("plane_rate_bounds", &sk::Guard::plane_rate_bounds)
      .def("clip", &sk::Guard::clip, py::arg("q_from"), py::arg("q_to"), py::arg("others") = Others{},
           py::call_guard<py::gil_scoped_release>())
      .def("speed_scale", &sk::Guard::speed_scale, py::arg("q"), py::arg("dq_step"), py::arg("dt"),
           py::arg("others") = Others{}, py::call_guard<py::gil_scoped_release>())
      .def("distances", &sk::Guard::distances, py::arg("q"), py::arg("others") = Others{})
      .def("capsules_at", &sk::Guard::capsules_at, py::arg("q"))
      .def("swept_capsules", &sk::Guard::swept_capsules, py::arg("q_from"), py::arg("q_to"));
}
