/*
 *
 * Copyright (c) 2023 Linhao Meng (Eindhoven University of Technology)
 * All rights reserved.
 *
 */

//pragma once is a non-standard but widely supported preprocessor directive designed to cause the current source file to be included only once in a single compilation
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "hdi/dimensionality_reduction/ctsne.h"

#include <cstdio>
#include <iostream>
#include <fstream>
#include <vector>

class cstsne {
    public:
        cstsne(
            int iterations = 1000,
            int perplexity = 30,
            int exaggeration_iter = 300
        ):_verbose(true), _iterations(iterations), _num_target_dimensions(2),_perplexity(perplexity), _exaggeration_iter(exaggeration_iter),_lambda(0.5),_alpha(0.5){};

        void clear() {
            tSNE.clear();
        }

        // update class probabilities
        void update_probabilities(
            py::array_t<float, py::array::c_style | py::array::forcecast> P) {
            std::cout << "update probabilities" << "\n";
            try {
                py::buffer_info P_info = P.request();

                _num_classes = P_info.shape[1];

                tSNE.setClassNum(_num_classes);
                std::cout << "classnum is" << P_info.shape[1] << "\n";
                auto ptr_c = static_cast<float*>(P_info.ptr);
                tSNE.setProbabilityData(ptr_c);

                // need to recompute P_s after updating probability data
                tSNE.initializeClassInfo(&classembedding);
            }
            catch (const std::exception& e) {
                std::cout << "Fatal error: " << e.what() << std::endl;
            }
        }

        // update projection 
        py::array_t<float, py::array::c_style> update_proj() {
            auto result = py::array_t<float>(0);
            try {
                // set exergation rate directly has effect on the learning rate so here we don't change that but just change when to remove it
                tSNE.resetRemoveExaggerationIter(0);

                std::cout << "grad descent tsne starting" << "\n";
                {
                    std::cout << "alpha|" << _alpha << "\n";
                    tSNE.initializeGradientDescent();
                    for (int iter = 0; iter < _iterations; ++iter) {
                        tSNE.doAnIteration(_alpha, 1., _lambda);
                        if (_verbose && iter % 100 == 0) {
                            std::cout << "Iter: " << iter << "\n";
                        }
                        if (tSNE.isconverged()) {
                            break;
                        }
                    }
                    if (_verbose) {
                        std::cout << "... done!\n";
                    }
                }
                std::cout << "grad descent tsne done" << "\n";

                auto size = (_num_data_points + _num_classes) * _num_target_dimensions;
                result = py::array_t<float>(size);
                py::buffer_info result_info = result.request();
                float* output = static_cast<float*>(result_info.ptr);

                auto data_d = embedding.getContainer().data();
                auto size_d = _num_data_points * _num_target_dimensions;
                for (decltype(size) i = 0; i < size_d; i++) {
                    output[i] = data_d[i];
                }

                auto data_c = classembedding.getContainer().data();
                for (decltype(size) i = size_d; i < size; i++) {
                    output[i] = data_c[i - size_d];
                }
                std::cout << "tsne done!\n";
            }
            catch (const std::exception& e) {
                std::cout << "Fatal error: " << e.what() << std::endl;
            }
            return result;
        }

        void set_class_embedding(py::array_t<float, py::array::c_style | py::array::forcecast> Yc) {
            auto Yc_loc = Yc;
            py::buffer_info Yc_info = Yc.request();

            auto ptr = static_cast<float*>(Yc_info.ptr);
            tSNE.setClassEmbeddingPosition(ptr);
        }

        void set_data_embedding(py::array_t<float, py::array::c_style | py::array::forcecast> Yd) {
            auto Yd_loc = Yd;
            py::buffer_info Yd_info = Yd.request();

            auto ptr = static_cast<float*>(Yd_info.ptr);
            tSNE.setDataEmbeddingPosition(ptr);
        }

        // tSNE transform and return results
        py::array_t<float, py::array::c_style> fit_transform(
            py::array_t<float, py::array::c_style | py::array::forcecast> X, std::optional<py::array_t<float, py::array::c_style | py::array::forcecast>> P){
            if (_verbose) {
                std::cout << "Iterations: " << _iterations << "\n";
                std::cout << "Target dimensions: " << _num_target_dimensions << "\n";
                std::cout << "Perplexity: " << _perplexity << "\n";
                std::cout << "Exaggeration iter.: " << _exaggeration_iter << "\n";
            }

            auto result = py::array_t<float>(0);
            try {

                auto X_loc = X;
                py::buffer_info X_info = X_loc.request();

                _num_data_points = X_info.shape[0];
                _num_dimensions = X_info.shape[1];

                tSNE.resetRemoveExaggerationIter(_exaggeration_iter);
                tSNE.setPerplexity(_perplexity);

                tSNE.setDimensionality(_num_dimensions);

                auto ptr = static_cast<float*>(X_info.ptr);
                std::cout << "size" << tSNE.size() << "\n";
                for (int d = 0; d < _num_data_points; d++) {
                    tSNE.addDataPoint(ptr + d * _num_dimensions);
                }
                tSNE.initializeDataInfo(&embedding);


                py::buffer_info P_info = P.value().request();
                _num_classes = P_info.shape[1];
                std::cout <<"prediction shape:"<< P_info.shape[0] << P_info.shape[1] << "\n";

                if (_num_classes > 0) {
                    tSNE.setClassNum(_num_classes);

                    auto ptr_c = static_cast<float*>(P_info.ptr);
                    tSNE.setProbabilityData(ptr_c);

                    tSNE.initializeClassInfo(&classembedding);
                }

                std::cout << "grad descent tsne starting" << "\n";
                {

                    tSNE.initializeGradientDescent();
                    std::cout << "alpha|" << _alpha <<  "lamda|"<<_lambda<< "\n";

                    for (int iter = 0; iter < _iterations; ++iter) {
                        tSNE.doAnIteration(_alpha,1., _lambda);
                        if (_verbose && iter%100==0) {
                            std::cout << "Iter: " << iter << "\n";
                        }
                        if (tSNE.isconverged()) {
                            break;
                        }
                    }
                    if (_verbose) {
                        std::cout << "... done!\n";
                    }
                }
                std::cout << "grad descent tsne done" << "\n";
                //tSNE.reset();

                auto size = (_num_data_points + _num_classes) * _num_target_dimensions;
                result = py::array_t<float>(size);
                py::buffer_info result_info = result.request();
                float *output = static_cast<float *>(result_info.ptr);

                auto data_d = embedding.getContainer().data();
                auto size_d = _num_data_points * _num_target_dimensions;
                for (decltype(size) i = 0; i < size_d; i++) {
                    output[i] = data_d[i];
                }

                auto data_c = classembedding.getContainer().data();
                for (decltype(size) i = size_d; i < size; i++) {
                    output[i] = data_c[i- size_d];
                }
                std::cout << "tsne done!\n";
            }
            catch (const std::exception& e) {
                std::cout << "Fatal error: " << e.what() << std::endl;
            }
            return result;
        };

        bool get_verbose() { return _verbose;  }
        int get_num_target_dimensions() { return _num_target_dimensions; }
        int get_iterations() { return _iterations; }
        int get_perplexity() { return _perplexity;  }
        int get_exaggeration_iter() { return _exaggeration_iter; }
        void set_iterations(int iterations) {
            _iterations = iterations;
            return;
        }
        void set_alpha(double alpha) {
            _alpha = alpha;
            return;
        }
        void set_lambda(double lambda) {
            _lambda = lambda;
            return;
        }
        void set_metric(std::string metric) {
            tSNE.setmetric(metric);
            return;
        }
        double getKL1() {
            return tSNE.computeKullbackLeiblerDivergence();
        }

        double getKL2() {
            return tSNE.computeKullbackLeiblerDivergenceClass();
        }

    private:
        typedef float scalar_type;

        hdi::dr::TSNE<scalar_type> tSNE;
        hdi::data::Embedding<scalar_type> embedding;
        hdi::data::Embedding<scalar_type> classembedding;

        int _num_data_points;
        int _num_dimensions;
        int _num_classes;

        double _alpha;
        double _lambda;

        bool _verbose;
        int _iterations;
        int _perplexity;
        int _num_target_dimensions;
        int _exaggeration_iter;
};
