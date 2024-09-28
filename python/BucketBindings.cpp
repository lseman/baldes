/**
 * @file BucketBindings.cpp
 * @brief Python bindings for the BucketGraph, VRPJob, Label, PSTEPDuals, and BucketOptions classes using pybind11.
 *
 * This file defines the Python module `baldes` and exposes the following classes:
 * - VRPJob: Represents a job in the VRP (Vehicle Routing Problem).
 * - Label: Represents a label used in the labeling algorithm.
 * - BucketGraph: Represents the graph structure used in the bucket-based VRP solver.
 * - PSTEPDuals: Represents dual values used in the PSTEP algorithm.
 * - BucketOptions: Represents configuration options for the bucket-based VRP solver.
 *
 * Each class is exposed with its constructors, member variables, and member functions.
 * Conditional compilation is used to expose additional fields and methods based on defined macros.
 */
#include "../bucket/BucketGraph.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals; // Enables _a suffix for named arguments

PYBIND11_MODULE(baldes, m) {
    py::class_<VRPJob>(m, "VRPJob")
        .def(py::init<>())                                  // Default constructor
        .def(py::init<int, int, int, int, double>())        // Constructor with multiple arguments
        .def_readwrite("x", &VRPJob::x)                     // Expose x
        .def_readwrite("y", &VRPJob::y)                     // Expose y
        .def_readwrite("id", &VRPJob::id)                   // Expose id
        .def_readwrite("start_time", &VRPJob::start_time)   // Expose start_time
        .def_readwrite("end_time", &VRPJob::end_time)       // Expose end_time
        .def_readwrite("duration", &VRPJob::duration)       // Expose duration
        .def_readwrite("cost", &VRPJob::cost)               // Expose cost
        .def_readwrite("demand", &VRPJob::demand)           // Expose demand
        .def_readwrite("consumption", &VRPJob::consumption) // Expose consumption (vector of double)
        .def_readwrite("lb", &VRPJob::lb)                   // Expose lb (vector of int)
        .def_readwrite("ub", &VRPJob::ub)                   // Expose ub (vector of int)
        .def("set_location", &VRPJob::set_location)         // Expose set_location function
        .def("add_arc", py::overload_cast<int, int, std::vector<double>, double, bool>(
                            &VRPJob::add_arc))  // Expose add_arc with one version
        .def("clear_arcs", &VRPJob::clear_arcs) // Expose clear_arcs function
        .def("sort_arcs", &VRPJob::sort_arcs);  // Expose sort_arcs function

    py::class_<Label>(m, "Label")
        .def(py::init<>()) // Default constructor
        .def(py::init<int, double, const std::vector<double> &, int, int>(), py::arg("vertex"), py::arg("cost"),
             py::arg("resources"), py::arg("pred"), py::arg("job_id"))
        .def(py::init<int, double, const std::vector<double> &, int>(), py::arg("vertex"), py::arg("cost"),
             py::arg("resources"), py::arg("pred"))
        .def_readwrite("is_extended", &Label::is_extended)
        .def_readwrite("vertex", &Label::vertex)
        .def_readwrite("cost", &Label::cost)
        .def_readwrite("real_cost", &Label::real_cost)
        .def_readwrite("resources", &Label::resources)
        .def_readwrite("jobs_covered", &Label::jobs_covered)
        .def_readwrite("job_id", &Label::job_id)
        .def_readwrite("parent", &Label::parent)
        .def_readwrite("visited_bitmap", &Label::visited_bitmap)
#ifdef UNREACHABLE_DOMINANCE
        .def_readwrite("unreachable_bitmap", &Label::unreachable_bitmap)
#endif
#ifdef SRC3
        .def_readwrite("SRCmap", &Label::SRCmap)
#endif
#ifdef SRC
        .def_readwrite("SRCmap", &Label::SRCmap)
#endif
        .def("set_extended", &Label::set_extended)
        .def("visits", &Label::visits)
        .def("reset", &Label::reset)
        .def("addJob", &Label::addJob)
        .def("initialize", &Label::initialize)
        .def("__repr__", [](const Label &label) {
            return "<bucket_graph.Label vertex=" + std::to_string(label.vertex) +
                   " cost=" + std::to_string(label.cost) + ">";
        });

    py::class_<BucketGraph>(m, "BucketGraph")
        .def(py::init<>()) // Default constructor
        .def(py::init<const std::vector<VRPJob> &, int, int>(), "jobs"_a, "time_horizon"_a, "bucket_interval"_a)
        .def("setup", &BucketGraph::setup)                            // Bind the setup method
        .def("redefine", &BucketGraph::redefine, "bucket_interval"_a) // Bind redefine method
        .def("solve", &BucketGraph::solve)                            // Bind solve method
        .def("set_adjacency_list", &BucketGraph::set_adjacency_list)  // Bind adjacency list setup
        .def("get_jobs", &BucketGraph::getJobs)                       // Get the jobs in the graph
        .def("print_statistics", &BucketGraph::print_statistics)      // Print stats
        .def("set_duals", &BucketGraph::setDuals, "duals"_a)          // Set dual values
        .def("set_distance_matrix", &BucketGraph::set_distance_matrix, "distance_matrix"_a,
             "n_ng"_a = 8)                           // Set distance matrix
        .def("reset_pool", &BucketGraph::reset_pool) // Reset the label pools
        .def("phaseOne", &BucketGraph::run_labeling_algorithms<Stage::One, Full::Partial>)
        .def("phaseTwo", &BucketGraph::run_labeling_algorithms<Stage::Two, Full::Partial>)
        .def("phaseThree", &BucketGraph::run_labeling_algorithms<Stage::Three, Full::Partial>)
#ifdef PSTEP
        .def("setArcDuals", &BucketGraph::setArcDuals, "duals"_a)
        .def("solvePSTEP", &BucketGraph::solvePSTEP)
#endif
        .def("setOptions", &BucketGraph::setOptions, "options"_a)
        .def("phaseFour", &BucketGraph::run_labeling_algorithms<Stage::Four, Full::Partial>);

    // Expose PSTEPDuals class
    py::class_<PSTEPDuals>(m, "PSTEPDuals")
        .def(py::init<>())                                                                   // Default constructor
        .def("set_arc_dual_values", &PSTEPDuals::setArcDualValues, "values"_a)               // Set arc dual values
        .def("set_threetwo_dual_values", &PSTEPDuals::setThreeTwoDualValues, "values"_a)     // Set node dual values
        .def("set_threethree_dual_values", &PSTEPDuals::setThreeThreeDualValues, "values"_a) // Set node dual values
        .def("clear_dual_values", &PSTEPDuals::clearDualValues)                              // Clear all dual values
        .def("__repr__", [](const PSTEPDuals &pstepDuals) { return "<PSTEPDuals with arc and node dual values>"; });

    py::class_<BucketOptions>(m, "BucketOptions")
        .def(py::init<>())                                             // Default constructor
        .def_readwrite("depot", &BucketOptions::depot)                 // Expose depot field
        .def_readwrite("end_depot", &BucketOptions::end_depot)         // Expose end_depot field
        .def_readwrite("max_path_size", &BucketOptions::max_path_size) // Expose max_path_size field
        .def("__repr__", [](const BucketOptions &options) {
            return "<BucketOptions depot=" + std::to_string(options.depot) +
                   " end_depot=" + std::to_string(options.end_depot) +
                   " max_path_size=" + std::to_string(options.max_path_size) + ">";
        });
}
