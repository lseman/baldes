/**
 * @file BucketBindings.cpp
 * @brief Python bindings for the BucketGraph, VRPNode, Label, PSTEPDuals, and BucketOptions classes using pybind11.
 *
 * This file defines the Python module `baldes` and exposes the following classes:
 * - VRPNode: Represents a node in the VRP (Vehicle Routing Problem).
 * - Label: Represents a label used in the labeling algorithm.
 * - BucketGraph: Represents the graph structure used in the bucket-based VRP solver.
 * - PSTEPDuals: Represents dual values used in the PSTEP algorithm.
 * - BucketOptions: Represents configuration options for the bucket-based VRP solver.
 *
 * Each class is exposed with its constructors, member variables, and member functions.
 * Conditional compilation is used to expose additional fields and methods based on defined macros.
 */
#include "Definitions.h"
#include "bucket/BucketGraph.h"
#include "bucket/BucketSolve.h"
#include "bucket/BucketUtils.h"

#include "Arc.h"
#include "Dual.h"
#include "Label.h"
#include "VRPNode.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef GUROBI
#include "gurobi_c++.h"
#include "gurobi_c.h"
#endif

namespace py = pybind11;
using namespace pybind11::literals; // Enables _a suffix for named arguments

PYBIND11_MODULE(pybaldes, m) {
    py::class_<VRPNode>(m, "VRPNode")
        .def(py::init<>())                                   // Default constructor
        .def(py::init<int, int, int, int, double>())         // Constructor with multiple arguments
        .def_readwrite("x", &VRPNode::x)                     // Expose x
        .def_readwrite("y", &VRPNode::y)                     // Expose y
        .def_readwrite("id", &VRPNode::id)                   // Expose id
        .def_readwrite("start_time", &VRPNode::start_time)   // Expose start_time
        .def_readwrite("end_time", &VRPNode::end_time)       // Expose end_time
        .def_readwrite("duration", &VRPNode::duration)       // Expose duration
        .def_readwrite("cost", &VRPNode::cost)               // Expose cost
        .def_readwrite("demand", &VRPNode::demand)           // Expose demand
        .def_readwrite("consumption", &VRPNode::consumption) // Expose consumption (vector of double)
        .def_readwrite("lb", &VRPNode::lb)                   // Expose lb (vector of int)
        .def_readwrite("ub", &VRPNode::ub)                   // Expose ub (vector of int)
        .def_readwrite("mtw_lb", &VRPNode::mtw_lb)           // Expose mtw_lb (vector of int)
        .def_readwrite("mtw_ub", &VRPNode::mtw_ub)           // Expose mtw_ub (vector of int)
        .def_readwrite("identifier", &VRPNode::identifier)   // Expose identifier
        .def("set_location", &VRPNode::set_location)         // Expose set_location function
        .def("add_arc", py::overload_cast<int, int, std::vector<double>, double, bool>(
                            &VRPNode::add_arc))  // Expose add_arc with one version
        .def("clear_arcs", &VRPNode::clear_arcs) // Expose clear_arcs function
        .def("sort_arcs", &VRPNode::sort_arcs);  // Expose sort_arcs function

    py::class_<Label>(m, "Label")
        .def(py::init<>()) // Default constructor
        .def(py::init<int, double, const std::vector<double> &, int, int>(), py::arg("vertex"), py::arg("cost"),
             py::arg("resources"), py::arg("pred"), py::arg("node_id"))
        .def(py::init<int, double, const std::vector<double> &, int>(), py::arg("vertex"), py::arg("cost"),
             py::arg("resources"), py::arg("pred"))
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

            SRC_MODE_BLOCK(.def_readwrite("SRCmap", &Label::SRCmap))
        .def("set_extended", &Label::set_extended)
        .def("visits", &Label::visits)
        .def("reset", &Label::reset)
        .def("addNode", &Label::addNode)
        .def("initialize", &Label::initialize)
        .def("__repr__", [](const Label &label) {
            return "<bucket_graph.Label vertex=" + std::to_string(label.vertex) +
                   " cost=" + std::to_string(label.cost) + ">";
        });

    py::class_<BucketGraph>(m, "BucketGraph")
        .def(py::init<>()) // Default constructor
        .def(py::init<const std::vector<VRPNode> &, int, int>(), "nodes"_a, "time_horizon"_a, "bucket_interval"_a)
        .def("setup", &BucketGraph::setup)                            // Bind the setup method
        .def("redefine", &BucketGraph::redefine, "bucket_interval"_a) // Bind redefine method
        .def("solve", &BucketGraph::solve<Symmetry::Asymmetric>, py::arg("arg0") = false,
             py::return_value_policy::reference)
        .def("extend_path", &BucketGraph::extend_path, "path"_a, "resources"_a,
             py::return_value_policy::reference) // Bind extend_path method

        .def("set_adjacency_list", &BucketGraph::set_adjacency_list<Symmetry::Asymmetric>) // Bind adjacency list setup
        .def("get_nodes", &BucketGraph::getNodes)                                          // Get the nodes in the graph
        .def("print_statistics", &BucketGraph::print_statistics)                           // Print stats
        .def("set_duals", &BucketGraph::setDuals, "duals"_a)                               // Set dual values
        .def("set_distance_matrix", &BucketGraph::set_distance_matrix, "distance_matrix"_a,
             "n_ng"_a = 8)                           // Set distance matrix
        .def("reset_pool", &BucketGraph::reset_pool) // Reset the label pools
        .def("phaseOne", &BucketGraph::run_labeling_algorithms<Stage::One, Full::Partial>)
        .def("phaseTwo", &BucketGraph::run_labeling_algorithms<Stage::Two, Full::Partial>)
        .def("phaseThree", &BucketGraph::run_labeling_algorithms<Stage::Three, Full::Partial>)
        // .def("setPSTEPDuals", &BucketGraph::setPSTEPduals, "duals"_a)
        .def("solvePSTEP_by_MTZ", &BucketGraph::solvePSTEP_by_MTZ)
        .def("solveTSPTW_by_MTZ", &BucketGraph::solveTSPTW_by_MTZ)
        // .def("solvePSTEP", &BucketGraph::solvePSTEP, py::return_value_policy::reference)
        .def("setOptions", &BucketGraph::setOptions, "options"_a)
        .def("setArcs", &BucketGraph::setManualArcs, "arcs"_a)
        .def("phaseFour", &BucketGraph::bi_labeling_algorithm<Stage::Eliminate, Symmetry::Asymmetric>,
             py::return_value_policy::reference)
        .def(
            "update_ng_neighbors",
            [](BucketGraph &self, const std::vector<std::tuple<int, int>> &conflicts) {
                // Convert Python tuples to C++ pairs
                std::vector<std::pair<size_t, size_t>> cpp_conflicts;
                cpp_conflicts.reserve(conflicts.size());
                for (const auto &conflict : conflicts) {
                    cpp_conflicts.emplace_back(std::get<0>(conflict), std::get<1>(conflict));
                }
                self.update_neighborhoods(cpp_conflicts);
            },
            "Update NG neighborhoods based on conflicts")
        .def("get_neighborhood_size", &BucketGraph::get_neighborhood_size, "Get size of neighborhood for given node")
        .def("get_neighbors", &BucketGraph::get_neighbors, "Get list of neighbors for given node")
        .def("is_in_neighborhood", &BucketGraph::is_in_neighborhood, "Check if node j is in node i's neighborhood")
        .def(
            "set_deleted_arcs",
            [](BucketGraph &self, const std::vector<std::tuple<int, int>> &arcs) {
                // Convert Python tuples to ArcList entries to mark as deleted
                std::vector<std::pair<int, int>> deleted_arcs;
                for (const auto &arc : arcs) { deleted_arcs.emplace_back(std::get<0>(arc), std::get<1>(arc)); }
                self.set_deleted_arcs(deleted_arcs);
            },
            "Set arcs that should be forbidden/deleted from the graph")
        .def("reset_fixed_arcs", &BucketGraph::reset_fixed, "Reset all fixed arcs in the graph");

    // Expose PSTEPDuals class
    py::class_<PSTEPDuals>(m, "PSTEPDuals")
        .def(py::init<>())                                                                   // Default constructor
        .def("set_arc_dual_values", &PSTEPDuals::setArcDualValues, "values"_a)               // Set arc dual values
        .def("set_threetwo_dual_values", &PSTEPDuals::setThreeTwoDualValues, "values"_a)     // Set node dual values
        .def("set_threethree_dual_values", &PSTEPDuals::setThreeThreeDualValues, "values"_a) // Set node dual values
        .def("clear_dual_values", &PSTEPDuals::clearDualValues)                              // Clear all dual values
        .def("__repr__", [](const PSTEPDuals &pstepDuals) { return "<PSTEPDuals with arc and node dual values>"; });

    py::class_<BucketOptions>(m, "BucketOptions")
        .def(py::init<>())                                                   // Default constructor
        .def_readwrite("depot", &BucketOptions::depot)                       // Expose depot field
        .def_readwrite("end_depot", &BucketOptions::end_depot)               // Expose end_depot field
        .def_readwrite("max_path_size", &BucketOptions::max_path_size)       // Expose max_path_size field
        .def_readwrite("min_path_size", &BucketOptions::min_path_size)       // Expose min_path_size field
        .def_readwrite("main_resources", &BucketOptions::main_resources)     // Expose main_resources field
        .def_readwrite("resources", &BucketOptions::resources)               // Expose resources field
        .def_readwrite("size", &BucketOptions::size)                         // Expose size field
        .def_readwrite("three_two_sign", &BucketOptions::three_two_sign)     // Expose three_two_sign field
        .def_readwrite("three_three_sign", &BucketOptions::three_three_sign) // Expose three_three_sign field
        .def_readwrite("three_five_sign", &BucketOptions::three_five_sign)   // Expose three_five_sign field
        .def_readwrite("pstep", &BucketOptions::pstep)                       // Expose pstep field
        .def_readwrite("resource_type", &BucketOptions::resource_type)       // Expose resource_type field
        .def_readwrite("bucket_fixing", &BucketOptions::bucket_fixing)       // Expose bucket_fixing field
        .def("__repr__", [](const BucketOptions &options) {
            return "<BucketOptions depot=" + std::to_string(options.depot) +
                   " end_depot=" + std::to_string(options.end_depot) +
                   " max_path_size=" + std::to_string(options.max_path_size) + ">";
        });

    py::class_<Arc>(m, "Arc")
        .def(py::init<int, int, const std::vector<double> &, double>())
        .def(py::init<int, int, const std::vector<double> &, double, bool>())
        .def(py::init<int, int, const std::vector<double> &, double, double>())
        .def_readonly("from", &Arc::from)
        .def_readonly("to", &Arc::to)
        .def_readonly("resource_increment", &Arc::resource_increment)
        .def_readonly("cost_increment", &Arc::cost_increment)
        .def_readonly("fixed", &Arc::fixed)
        .def_readonly("priority", &Arc::priority);

    py::class_<ArcList>(m, "ArcList")
        .def(py::init<>())
        .def("add_connections", &ArcList::add_connections, py::arg("connections"),
             py::arg("default_resource_increment") = std::vector<double>{1.0}, py::arg("default_cost_increment") = 0.0,
             py::arg("default_priority") = 1.0)
        .def("get_arcs", &ArcList::get_arcs);
}
