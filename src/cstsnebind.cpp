/*
 *
 * Copyright (c) 2023 Linhao Meng (Eindhoven University of Technology)
 * All rights reserved.
 *
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // automatic conversion of STL to list, set, tuple, dict
#include <pybind11/stl_bind.h>

#include "cstsne.h"

#include <string>
#include <functional>
#include <tuple>

namespace py = pybind11;

PYBIND11_MODULE(cstsne, m) {
    py::class_<cstsne>(m, "cstsne")
        .def(py::init<int, int, int>())
        .def("fit_transform", &cstsne::fit_transform)
        .def("update_proj", &cstsne::update_proj)
        .def("set_probabilities", &cstsne::update_probabilities)
        .def("setClassEmbedding", &cstsne::set_class_embedding)
        .def("setDataEmbedding", &cstsne::set_data_embedding)
        .def("set_iterations", &cstsne::set_iterations)
        .def("set_alpha", &cstsne::set_alpha)
        .def("set_lambda", &cstsne::set_lambda)
        .def("set_metric", &cstsne::set_metric)
        .def("getKL1", &cstsne::getKL1)
        .def("getKL2", &cstsne::getKL2)
        .def("clear", &cstsne::clear);
};