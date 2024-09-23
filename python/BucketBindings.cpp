#include "BucketGraph.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals; // Enables _a suffix for named arguments

PYBIND11_MODULE(bucket_graph, m) {
    py::class_<VRPJob>(m, "VRPJob")
        .def(py::init<>())  // Default constructor
        .def(py::init<int, int, int, int, double>())  // Constructor with multiple arguments
        .def_readwrite("x", &VRPJob::x)  // Expose x
        .def_readwrite("y", &VRPJob::y)  // Expose y
        .def_readwrite("id", &VRPJob::id)  // Expose id
        .def_readwrite("start_time", &VRPJob::start_time)  // Expose start_time
        .def_readwrite("end_time", &VRPJob::end_time)  // Expose end_time
        .def_readwrite("duration", &VRPJob::duration)  // Expose duration
        .def_readwrite("cost", &VRPJob::cost)  // Expose cost
        .def_readwrite("demand", &VRPJob::demand)  // Expose demand
        .def_readwrite("consumption", &VRPJob::consumption)  // Expose consumption (vector of double)
        .def_readwrite("lb", &VRPJob::lb)  // Expose lb (vector of int)
        .def_readwrite("ub", &VRPJob::ub)  // Expose ub (vector of int)
        .def("set_location", &VRPJob::set_location)  // Expose set_location function
        .def("add_arc", py::overload_cast<int, int, std::vector<double>, double, bool>(&VRPJob::add_arc))  // Expose add_arc with one version
        .def("clear_arcs", &VRPJob::clear_arcs)  // Expose clear_arcs function
        .def("sort_arcs", &VRPJob::sort_arcs);  // Expose sort_arcs function

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
        .def("run_labeling_algorithms",
             &BucketGraph::run_labeling_algorithms<Stage::Two,
                                                   Full::Partial>) // Example of exposing a templated function
        ;
}
