#include "Arc.h"
#include "Dual.h"
#include "Label.h"
#include "VRPNode.h"
#include "bucket/BucketGraph.h"
#include "bucket/BucketSolve.h"
#include "bucket/BucketUtils.h"
#include <nanobind/nanobind.h>

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

namespace nb = nanobind;
using namespace nanobind::literals;

NB_MODULE(pybaldes, m) {
    nb::class_<VRPNode>(m, "VRPNode")
        .def(nb::init<>())
        .def(nb::init<int, int, int, int, double>())
        .def_readwrite("x", &VRPNode::x)
        .def_readwrite("y", &VRPNode::y)
        .def_readwrite("id", &VRPNode::id)
        .def_readwrite("start_time", &VRPNode::start_time)
        .def_readwrite("end_time", &VRPNode::end_time)
        .def_readwrite("duration", &VRPNode::duration)
        .def_readwrite("cost", &VRPNode::cost)
        .def_readwrite("demand", &VRPNode::demand)
        .def_readwrite("consumption", &VRPNode::consumption)
        .def_readwrite("lb", &VRPNode::lb)
        .def_readwrite("ub", &VRPNode::ub)
        .def("set_location", &VRPNode::set_location)
        .def("add_arc", nb::overload_cast<int, int, std::vector<double>, double, bool>(&VRPNode::add_arc))
        .def("clear_arcs", &VRPNode::clear_arcs)
        .def("sort_arcs", &VRPNode::sort_arcs);

    nb::class_<Label>(m, "Label")
        .def(nb::init<>())
        .def(nb::init<int, double, const std::vector<double> &, int, int>(), "vertex"_a, "cost"_a, "resources"_a,
             "pred"_a, "node_id"_a)
        .def(nb::init<int, double, const std::vector<double> &, int>(), "vertex"_a, "cost"_a, "resources"_a, "pred"_a)
        .def_readwrite("is_extended", &Label::is_extended)
        .def_readwrite("vertex", &Label::vertex)
        .def_readwrite("cost", &Label::cost)
        .def_readwrite("real_cost", &Label::real_cost)
        .def_readwrite("resources", &Label::resources)
        .def_readwrite("nodes_covered", &Label::nodes_covered)
        .def_readwrite("node_id", &Label::node_id)
        .def_readwrite("parent", &Label::parent)
        .def_readwrite("visited_bitmap", &Label::visited_bitmap)
#ifdef UNREACHABLE_DOMINANCE
        .def_readwrite("unreachable_bitmap", &Label::unreachable_bitmap)
#endif
        .def("set_extended", &Label::set_extended)
        .def("visits", &Label::visits)
        .def("reset", &Label::reset)
        .def("addNode", &Label::addNode)
        .def("initialize", &Label::initialize)
        .def("__repr__", [](const Label &label) {
            return "<bucket_graph.Label vertex=" + std::to_string(label.vertex) +
                   " cost=" + std::to_string(label.cost) + ">";
        });

    nb::class_<BucketGraph>(m, "BucketGraph")
        .def(nb::init<>())
        .def(nb::init<const std::vector<VRPNode> &, int, int>(), "nodes"_a, "time_horizon"_a, "bucket_interval"_a)
        .def("setup", &BucketGraph::setup)
        .def("redefine", &BucketGraph::redefine, "bucket_interval"_a)
        .def("solve", &BucketGraph::solve, nb::arg("arg0") = false, nb::return_value_policy::reference)
        .def("set_adjacency_list", &BucketGraph::set_adjacency_list)
        .def("get_nodes", &BucketGraph::getNodes)
        .def("print_statistics", &BucketGraph::print_statistics)
        .def("set_duals", &BucketGraph::setDuals, "duals"_a)
        .def("set_distance_matrix", &BucketGraph::set_distance_matrix, "distance_matrix"_a, "n_ng"_a = 8)
        .def("reset_pool", &BucketGraph::reset_pool)
        .def("phaseOne", &BucketGraph::run_labeling_algorithms<Stage::One, Full::Partial>)
        .def("phaseTwo", &BucketGraph::run_labeling_algorithms<Stage::Two, Full::Partial>)
        .def("phaseThree", &BucketGraph::run_labeling_algorithms<Stage::Three, Full::Partial>)
#ifdef PSTEP
        .def("solvePSTEP", &BucketGraph::solvePSTEP, nb::return_value_policy::reference)
#endif
        .def("setOptions", &BucketGraph::setOptions, "options"_a)
        .def("setArcs", &BucketGraph::setManualArcs, "arcs"_a)
        .def("phaseFour", &BucketGraph::run_labeling_algorithms<Stage::Four, Full::Partial>);

    nb::class_<PSTEPDuals>(m, "PSTEPDuals")
        .def(nb::init<>())
        .def("set_arc_dual_values", &PSTEPDuals::setArcDualValues, "values"_a)
        .def("set_threetwo_dual_values", &PSTEPDuals::setThreeTwoDualValues, "values"_a)
        .def("set_threethree_dual_values", &PSTEPDuals::setThreeThreeDualValues, "values"_a)
        .def("clear_dual_values", &PSTEPDuals::clearDualValues)
        .def("__repr__", [](const PSTEPDuals &pstepDuals) { return "<PSTEPDuals with arc and node dual values>"; });

    nb::class_<BucketOptions>(m, "BucketOptions")
        .def(nb::init<>())
        .def_readwrite("depot", &BucketOptions::depot)
        .def_readwrite("end_depot", &BucketOptions::end_depot)
        .def_readwrite("max_path_size", &BucketOptions::max_path_size)
        .def_readwrite("main_resources", &BucketOptions::main_resources)
        .def_readwrite("resources", &BucketOptions::resources)
        .def_readwrite("size", &BucketOptions::size)
        .def_readwrite("resource_disposability", &BucketOptions::resource_disposability)
        .def("__repr__", [](const BucketOptions &options) {
            return "<BucketOptions depot=" + std::to_string(options.depot) +
                   " end_depot=" + std::to_string(options.end_depot) +
                   " max_path_size=" + std::to_string(options.max_path_size) + ">";
        });

    nb::class_<Arc>(m, "Arc")
        .def(nb::init<int, int, const std::vector<double> &, double>())
        .def(nb::init<int, int, const std::vector<double> &, double, bool>())
        .def(nb::init<int, int, const std::vector<double> &, double, double>())
        .def_readonly("from", &Arc::from)
        .def_readonly("to", &Arc::to)
        .def_readonly("resource_increment", &Arc::resource_increment)
        .def_readonly("cost_increment", &Arc::cost_increment)
        .def_readonly("fixed", &Arc::fixed)
        .def_readonly("priority", &Arc::priority);

    nb::class_<ArcList>(m, "ArcList")
        .def(nb::init<>())
        .def("add_connections", &ArcList::add_connections, "connections"_a,
             "default_resource_increment"_a = std::vector<double>{1.0}, "default_cost_increment"_a = 0.0,
             "default_priority"_a = 1.0)
        .def("get_arcs", &ArcList::get_arcs);
}
