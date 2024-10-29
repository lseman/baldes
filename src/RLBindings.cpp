#include "HighOrderSRC.h"
#include <pybind11/functional.h> // Needed to expose std::function
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(cut_gen, m) {
    py::class_<R1c>(m, "R1c")
        .def_readwrite("info_r1c", &R1c::info_r1c)
        .def_readwrite("rhs", &R1c::rhs)
        .def_readwrite("arc_mem", &R1c::arc_mem);

    py::class_<Rank1MultiLabel>(m, "Rank1MultiLabel")
        .def(py::init<std::vector<int>, std::vector<int>, int, double, char>(), py::arg("c"), py::arg("w_no_c"),
             py::arg("plan_idx"), py::arg("vio"), py::arg("search_dir"))
        .def_readwrite("c", &Rank1MultiLabel::c)
        .def_readwrite("w_no_c", &Rank1MultiLabel::w_no_c)
        .def_readwrite("plan_idx", &Rank1MultiLabel::plan_idx)
        .def_readwrite("vio", &Rank1MultiLabel::vio)
        .def_readwrite("search_dir", &Rank1MultiLabel::search_dir);

    py::class_<HighDimCutsGenerator>(m, "HighDimCutsGenerator")
        .def(py::init<int, int, double>(), py::arg("dim"), py::arg("maxRowRank"), py::arg("tolerance"))

        .def("get_high_dim_cuts", &HighDimCutsGenerator::getHighDimCuts)

        // Other methods from the class, e.g., initialize, set_nodes, etc.
        .def("initialize", &HighDimCutsGenerator::initialize, py::arg("routes"))
        .def("set_nodes", &HighDimCutsGenerator::setNodes, py::arg("nodes"))
        .def("get_cuts", &HighDimCutsGenerator::getCuts)
        .def("print_cuts", &HighDimCutsGenerator::printCuts);
}
