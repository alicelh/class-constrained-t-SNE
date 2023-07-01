/*
 *
 * Copyright (c) 2014, Nicola Pezzotti (Delft University of Technology)
 * All rights reserved.
 * Modified work Copyright 2023 Linhao Meng (Eindhoven University of Technology)
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 *  notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimer in the
 *  documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *  must display the following acknowledgement:
 *  This product includes software developed by the Delft University of Technology.
 * 4. Neither the name of the Delft University of Technology nor the names of
 *  its contributors may be used to endorse or promote products derived from
 *  this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY NICOLA PEZZOTTI ''AS IS'' AND ANY EXPRESS
 * OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL NICOLA PEZZOTTI BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
 * OF SUCH DAMAGE.
 *
 */


#ifndef TSNE_INL
#define TSNE_INL

#include "ctsne.h"
#include "../utils/math_utils.h"
#include "../utils/log_helper_functions.h"

#include <time.h>
#include <cmath>
#include <iostream>

#ifdef __USE_GCD__
#include <dispatch/dispatch.h>
#endif


namespace hdi {
    namespace dr {
        /////////////////////////////////////////////////////////////////////////

        template <typename scalar_type>
        TSNE<scalar_type>::InitParams::InitParams() :
            _perplexity(30),
            _seed(0),
            _embedding_dimensionality(2),
            _minimum_gain_d(0.01),
            _eta(200),
            _eta_c(1),
            _momentum(0.5),
            _final_momentum(0.8),
            _mom_switching_iter(300),
            _exaggeration_factor(12.),
            _min_grad_norm(1.0e-7)
        {}

        /////////////////////////////////////////////////////////////////////////

        template <typename scalar_type>
        TSNE<scalar_type>::TSNE() :
            _initialized(false),
            _dimensionality(0),
            _classnum(0),
            _logger(nullptr),
            _metric("euclidean")
        {

        }

        template <typename scalar_type>
        typename TSNE<scalar_type>::data_handle_type TSNE<scalar_type>::addDataPoint(const scalar_type* ptr) {
            checkAndThrowLogic(!_initialized, "Class should be uninitialized to add a data-point");
            checkAndThrowLogic(_dimensionality > 0, "Invalid dimensionality");
            _high_dimensional_data.push_back(ptr);
            return static_cast<data_handle_type>(_high_dimensional_data.size() - 1);
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::reset() {
            _initialized = false;
            _iteration = 0;
            _classnum = 0;
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::clear() {
            _high_dimensional_data.clear();
            _probability_data.clear();
            _embedding->clear();
            _class_embedding->clear();
            _initialized = false;
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::getHighDimensionalDescriptor(scalar_vector_type& data_point, data_handle_type handle)const {
            data_point.resize(_dimensionality);
            for (int i = 0; i < _dimensionality; ++i) {
                data_point[i] = *(_high_dimensional_data[handle] + i);
            }
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::getEmbeddingPosition(scalar_vector_type& embedding_position, data_handle_type handle)const {
            embedding_position.resize(_init_params._embedding_dimensionality);
            for (int i = 0; i < _init_params._embedding_dimensionality; ++i) {
                embedding_position[i] = _embedding->getContainer()[handle * _init_params._embedding_dimensionality + i];
            }
        }


        /////////////////////////////////////////////////////////////////////////


        template <typename scalar_type>
        void TSNE<scalar_type>::initializeDataInfo(data::Embedding<scalar_type>* embedding) {
            utils::secureLog(_logger, "Initializing tSNE...");
            if (size() == 0) {
                throw std::logic_error("Cannot initialize an empty dataset");
            }
            {
                _embedding = embedding;
                int size_sq = size();
                size_sq *= size_sq;
                _P_d.resize(size_sq);
                _Q_d.resize(size_sq);
                _distances_squared.resize(size_sq);
                _embedding->resize(_init_params._embedding_dimensionality, size(), 0);
                _embedding_container = &_embedding->getContainer();
                _sigmas.resize(size());
                _init_params = _init_params;
            }

            //compute distances between data-points
            computeHighDimensionalDistances();
            //Compute gaussian distributions
            computeGaussianDistributions(_init_params._perplexity);
            //Compute High-dimensional distribution
            compute_P_d();


            //Initialize Embedding position
            initializeEmbeddingPosition(_init_params._seed);

            utils::secureLog(_logger, "Initialization complete!");
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::computeHighDimensionalDistances() {
            utils::secureLog(_logger, "Computing High-dimensional distances...");
            const int n = size();
#ifdef __USE_GCD__
            std::cout << "GCD dispatch, tsne_inl 165.\n";
            dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
            for (int j = 0; j < n; ++j) {
#endif //__USE_GCD__
                _distances_squared[j * n + j] = 0;
                for (int i = j + 1; i < n; ++i) {
                    if (_metric == "cosine") {
                        scalar_type res(utils::cosinesimilarity<scalar_type>(_high_dimensional_data[i], _high_dimensional_data[i] + _dimensionality, _high_dimensional_data[j], _high_dimensional_data[j] + _dimensionality));
                        _distances_squared[j * n + i] = (1. - res)* (1. - res);
                        _distances_squared[i * n + j] = (1. - res) * (1. - res);
                    }
                    else {
                        scalar_type res(utils::euclideanDistanceSquared<scalar_type>(_high_dimensional_data[i], _high_dimensional_data[i] + _dimensionality, _high_dimensional_data[j], _high_dimensional_data[j] + _dimensionality));
                        _distances_squared[j * n + i] = res;
                        _distances_squared[i * n + j] = res;
                    }
                }
            }
#ifdef __USE_GCD__
            );
#endif
            }

        template <typename scalar_type>
        void TSNE<scalar_type>::computeGaussianDistributions(double perplexity) {
            utils::secureLog(_logger, "Computing gaussian distributions...");
            const int n = size();
#ifdef __USE_GCD__
            std::cout << "GCD dispatch, tsne_inl 189.\n";
            dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
            for (int j = 0; j < n; ++j) {
#endif //__USE_GCD__
                const auto sigma = utils::computeGaussianDistributionWithFixedPerplexity<scalar_vector_type>(
                    _distances_squared.begin() + j * n,
                    _distances_squared.begin() + (j + 1) * n,
                    _P_d.begin() + j * n,
                    _P_d.begin() + (j + 1) * n,
                    perplexity,
                    200,
                    1e-5,
                    j
                    );
                _P_d[j * n + j] = 0.;
                _sigmas[j] = static_cast<scalar_type>(sigma);
            }
#ifdef __USE_GCD__
            );
#endif
            }

        template <typename scalar_type>
        void TSNE<scalar_type>::compute_P_d() {
            utils::secureLog(_logger, "Computing high-dimensional joint probability distribution...");
            const int n = size();
            //#pragma omp parallel for
            for (int j = 0; j < n; ++j) {
                for (int i = j + 1; i < n; ++i) {
                    const double v = (_P_d[j * n + i] + _P_d[i * n + j]) * 0.5 / n;
                    const double _v = v > MACHINE_EPSILON ? v : MACHINE_EPSILON;
                    _P_d[j * n + i] = static_cast<scalar_type>(_v);
                    _P_d[i * n + j] = static_cast<scalar_type>(_v);
                }
            }
        }


        template <typename scalar_type>
        void TSNE<scalar_type>::setDataEmbeddingPosition(const scalar_type* ptr) {
            for (int i = 0; i < _embedding->numDataPoints(); ++i) {
                _embedding->dataAt(i, 0) = *ptr++;
                _embedding->dataAt(i, 1) = *ptr++;
            }
        }


        template <typename scalar_type>
        void TSNE<scalar_type>::initializeEmbeddingPosition(int seed, double multiplier) {
            utils::secureLog(_logger, "Initializing the embedding...");
            if (seed < 0) {
                std::srand(static_cast<unsigned int>(time(NULL)));
            }
            else {
                std::srand(seed);
            }

            for (int i = 0; i < _embedding->numDataPoints(); ++i) {
                double x(0.);
                double y(0.);
                double radius(0.);
                do {
                    x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
                    y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
                    radius = (x * x) + (y * y);
                } while ((radius >= 1.0) || (radius == 0.0));

                radius = sqrt(-2 * log(radius) / radius);
                x *= radius * multiplier;
                y *= radius * multiplier;
                _embedding->dataAt(i, 0) = x;
                _embedding->dataAt(i, 1) = y;
            }
        }


        template <typename scalar_type>
        void TSNE<scalar_type>::doAnIteration(double alpha, double degrees_of_freedom, double lambda, double mult) {
            if (!_initialized) {
                throw std::logic_error("Cannot compute a gradient descent iteration on unitialized setting");
            }

            if (_iteration == _init_params._mom_switching_iter) {
                utils::secureLog(_logger, "Remove exaggeration and Switch to final momentum...");
            }

            //Compute Low-dimensional distribution
            compute_Q_d(degrees_of_freedom);

            if (getclassnum() > 0) {
                //Compute Low-dimensional distribution between instances to class glyphs
                compute_Q_s(degrees_of_freedom);
            }


            //Compute gradient of the KL function
            computeGradient((_iteration < _init_params._mom_switching_iter) ? _init_params._exaggeration_factor : 1., alpha, degrees_of_freedom, lambda);

            //Update Embedding
            updateTheEmbedding(mult);
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::compute_Q_d(double degrees_of_freedom) {
            const int n = size();
#ifdef __USE_GCD__
            std::cout << "GCD dispatch, tsne_inl 283.\n";
            dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
            for (int j = 0; j < n; ++j) {
#endif //__USE_GCD__
                _Q_d[j * n + j] = 0;
                for (int i = j + 1; i < n; ++i) {
                    const double euclidean_dist_sq(
                        utils::euclideanDistanceSquared<scalar_type>(
                            _embedding_container->begin() + j * _init_params._embedding_dimensionality,
                            _embedding_container->begin() + (j + 1) * _init_params._embedding_dimensionality,
                            _embedding_container->begin() + i * _init_params._embedding_dimensionality,
                            _embedding_container->begin() + (i + 1) * _init_params._embedding_dimensionality
                            )
                    );
                    const double v = pow(1. + euclidean_dist_sq/degrees_of_freedom, (degrees_of_freedom + 1.0) / -2.0);
                    const double _v = v > MACHINE_EPSILON ? v : MACHINE_EPSILON;
                    _Q_d[j * n + i] = static_cast<scalar_type>(_v);
                    _Q_d[i * n + j] = static_cast<scalar_type>(_v);
                }
            }
#ifdef __USE_GCD__
            );
#endif
            double sum_Q_d = 0;
            for (auto& v : _Q_d) {
                sum_Q_d += v;
            }
            sum_Q_d = sum_Q_d> MACHINE_EPSILON ? sum_Q_d : MACHINE_EPSILON;
            _normalization_Q_d = static_cast<scalar_type>(sum_Q_d);
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::computeGradient(double exaggeration, double alpha, double degrees_of_freedom, double lambda) {
            const int n = size();
            const int m = getclassnum();
            const int dim = _init_params._embedding_dimensionality;
            const double c = 2 * (degrees_of_freedom + 1.0) / degrees_of_freedom;

            //#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                for (int d = 0; d < dim; ++d) {
                    _gradient_d[i * dim + d] = 0;
                    double sum_positive_d(0.);
                    double sum_negative_d(0.);
                    double sum_positive_s(0.);
                    double sum_negative_s(0.);
                    double sum_distance(0.);
                    if (alpha < 1) {
                        for (int j = 0; j < n; ++j) {
                            const int idx = i * n + j;
                            const double distance((*_embedding_container)[i * dim + d] - (*_embedding_container)[j * dim + d]);
                            const double positive(_P_d[idx] * _Q_d[idx] * distance);
                            const double negative(_Q_d[idx] * _Q_d[idx] / _normalization_Q_d * distance);
                            sum_positive_d += positive;
                            sum_negative_d += negative;
                        }
                    }
                    if (alpha > 0) {
                        for (int j = 0; j < m; ++j) {
                            const int idx = i * m + j;
                            const double distance((*_embedding_container)[i * dim + d] - (*_class_embedding_container)[j * dim + d]);
                            const double positive(_P_s[idx] * _Q_s[idx] * distance);
                            const double negative(_Q_s[idx] * _Q_s[idx] / _normalization_Q_s[i] * distance);
                            sum_positive_s += positive / n;
                            sum_negative_s += negative / n;
                            sum_distance += (_P_s[idx] / n) * distance / m;
                        }
                    }
                    _gradient_d[i * dim + d] = static_cast<scalar_type>(c * (1. - alpha) * (exaggeration * sum_positive_d - sum_negative_d) + alpha * (2. * sum_distance * lambda + c / 2. * (sum_positive_s - sum_negative_s)));
                }
            }
            //#pragma omp parallel for
            for (int i = 0; i < m; ++i) {
                for (int d = 0; d < dim; ++d) {
                    _gradient_c[i * dim + d] = 0;
                    double sum_positive_s(0.);
                    double sum_negative_s(0.);
                    double sum_distance(0.);
                    for (int j = 0; j < n; ++j) {
                        const int idx = i + j * m;
                        const double distance((*_class_embedding_container)[i * dim + d] - (*_embedding_container)[j * dim + d]);
                        const double positive(_P_s[idx] * _Q_s[idx] * distance);
                        const double negative(_Q_s[idx] * _Q_s[idx] / _normalization_Q_s[j] * distance);
                        sum_positive_s += positive / n;
                        sum_negative_s += negative / n;
                        sum_distance += (_P_s[idx] / n) * distance  /m ;
                    }
                    _gradient_c[i * dim + d] = static_cast<scalar_type>( 2. * sum_distance * lambda + c / 2. * (sum_positive_s - sum_negative_s));
                }
            }
        }

        //temp
        template <typename T>
        T sign(T x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

        template <typename scalar_type>
        void TSNE<scalar_type>::updateTheEmbedding(double mult) {
            double norm_d = 0;
            double norm_c = 0;
            for (int i = 0; i < _gradient_d.size(); ++i) {
                norm_d += _gradient_d[i] * _gradient_d[i];
                _gain_d[i] = static_cast<scalar_type>(((_gradient_d[i]*_previous_gradient_d[i])<0) ? (_gain_d[i] + .2) : (_gain_d[i] * .8));
                if (_gain_d[i] < _init_params._minimum_gain_d) {
                    _gain_d[i] = static_cast<scalar_type>(_init_params._minimum_gain_d);
                }
                _gradient_d[i] *= _gain_d[i];
                _previous_gradient_d[i] = static_cast<scalar_type>(((_iteration < _init_params._mom_switching_iter) ? _init_params._momentum : _init_params._final_momentum) * _previous_gradient_d[i] - _init_params._eta * _gradient_d[i]);
                (*_embedding_container)[i] += _previous_gradient_d[i] * mult;

            }
            if (getclassnum()>0) {
                for (int i = 0; i < _gradient_c.size(); ++i) {
                    norm_c += _gradient_c[i] * _gradient_c[i];
                    _gain_c[i] = static_cast<scalar_type>(((_gradient_c[i] * _previous_gradient_c[i]) < 0) ? (_gain_c[i] + .2) : (_gain_c[i] * .8));
                    if (_gain_c[i] < _init_params._minimum_gain_d) {
                        _gain_c[i] = static_cast<scalar_type>(_init_params._minimum_gain_d);
                    }
                    _gradient_c[i] *= _gain_c[i];
                    _previous_gradient_c[i] = static_cast<scalar_type>(((_iteration < _init_params._mom_switching_iter) ? _init_params._momentum : _init_params._final_momentum) * _previous_gradient_c[i] - _init_params._eta_c * _gradient_c[i]);
                    (*_class_embedding_container)[i] += _previous_gradient_c[i] * mult;
                }
            }
            if (_iteration > _init_params._mom_switching_iter && _iteration % 50 == 0) {
                if (norm_c < _init_params._min_grad_norm) {
                    isclassconverged = true;
                    std::cout << "after iteration " << _iteration << "class is converged with norm_c " << norm_c << "\n";
                }
                if (norm_d < _init_params._min_grad_norm) {
                    isdataconverged = true;
                    std::cout << "after iteration " << _iteration << "data is converged with norm_d " << norm_d << "\n";
                }
            }
            ++_iteration;
        }

        template <typename scalar_type>
        double TSNE<scalar_type>::computeKullbackLeiblerDivergence() {
            double kl = 0;
            const int n = size();
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < n; ++i) {
                    if (i == j)
                        continue;
                    kl += _P_d[j * n + i] * std::log(_P_d[j * n + i] / (_Q_d[j * n + i] / _normalization_Q_d));
                }
            }
            return kl;
        }

        template <typename scalar_type>
        double TSNE<scalar_type>::computeKullbackLeiblerDivergenceClass() {
            double kl = 0;
            const int n = size();
            const int m = getclassnum();
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < m; ++i) {
                    kl += _P_s[j * m + i] * std::log((_P_s[j * m + i] > MACHINE_EPSILON ? _P_s[j * m + i] : MACHINE_EPSILON) / (_Q_s[j * m + i] / _normalization_Q_s[j])) / n;
                }
            }
            return kl;
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::initializeClassInfo(data::Embedding<scalar_type>* classembedding) {
            utils::secureLog(_logger, "Initializing class info...");
            if (getclassnum() == 0) {
                utils::secureLog(_logger, "Skip class initilization with 0 class");
                return;
            }
            {
                _class_embedding = classembedding;
                int n = size();
                int m = getclassnum();
                int size_sq = n * m;

                _P_s.resize(size_sq);
                _Q_s.resize(size_sq);
                _normalization_Q_s.resize(n);
            }

            //Compute high dimension distribution between class glyphs and data instances
            compute_P_s();

            initializeClassEmbeddingPosition(_init_params._seed);

            utils::secureLog(_logger, "Class initialization complete!");
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::compute_P_s() {
            const int n = size();
            const int m = getclassnum();
            for (int i = 0; i < n*m; ++i) {
                _P_s[i] = _probability_data[i] ;
            }
        }


        template <typename scalar_type>
        void TSNE<scalar_type>::compute_Q_s(double degrees_of_freedom) {
            const int n = size();
            const int m = getclassnum();

#ifdef __USE_GCD__
            std::cout << "GCD dispatch, tsne_inl 283.\n";
            dispatch_apply(n, dispatch_get_global_queue(0, 0), ^ (size_t j) {
#else
#pragma omp parallel for
            for (int j = 0; j < n; ++j) {
#endif //__USE_GCD__
                double sum_Q_s = 0;
                for (int i = 0; i < m; ++i) {
                    const double euclidean_dist_sq(
                        utils::euclideanDistanceSquared<scalar_type>(
                            _embedding_container->begin() + j * _init_params._embedding_dimensionality,
                            _embedding_container->begin() + (j + 1) * _init_params._embedding_dimensionality,
                            _class_embedding_container->begin() + i * _init_params._embedding_dimensionality,
                            _class_embedding_container->begin() + (i + 1) * _init_params._embedding_dimensionality
                            )
                    );
                    const double v = pow (1.0 + (euclidean_dist_sq / degrees_of_freedom), (degrees_of_freedom + 1.0) / -2.0);
                    const double _v = v > MACHINE_EPSILON ? v : MACHINE_EPSILON;
                    _Q_s[j * m + i] = static_cast<scalar_type>(_v);
                    sum_Q_s += _v;
                }
                _normalization_Q_s[j] = static_cast<scalar_type>((sum_Q_s ) > MACHINE_EPSILON? (sum_Q_s ): MACHINE_EPSILON);
            }
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::setProbabilityData(const scalar_type* ptr) {
            const int n = size();
            const int m = getclassnum();
            _probability_data.resize(n * m);
            for (int i = 0; i < n*m; ++i) {
                _probability_data[i] = *(ptr+i);
            }
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::initializeGradientDescent() {
            const int n = size();
            const int m = getclassnum();

            isclassconverged = false;
            isdataconverged = false;

            // Initialize learning rate
            const double learning_rate = n / _init_params._exaggeration_factor / 4.;
            _init_params._eta = learning_rate < 50 ? 50: learning_rate;
            _init_params._eta_c = _init_params._eta / n * m ;

            std::cout << "learning rate" << _init_params._eta << "|" << _init_params._eta_c << "\n";

            // Initialize gradient and gain
            _gradient_d.resize(n * _init_params._embedding_dimensionality, 0);
            _previous_gradient_d.resize(n * _init_params._embedding_dimensionality, 0);
            _gain_d.resize(n * _init_params._embedding_dimensionality, 1);
            

            if (m > 0) {
                std::cout << "initialize class gradient" << "\n";
                _gradient_c.resize(m * _init_params._embedding_dimensionality, 0);
                _previous_gradient_c.resize(m * _init_params._embedding_dimensionality, 0);
                _gain_c.resize(m * _init_params._embedding_dimensionality, 1);
            }

            _initialized = true;
            _iteration = 0;
        }

        template <typename scalar_type>
        void TSNE<scalar_type>::setClassEmbeddingPosition(const scalar_type* ptr) {
            for (int i = 0; i < _class_embedding->numDataPoints(); ++i) {
                _class_embedding->dataAt(i, 0) = *ptr++;
                _class_embedding->dataAt(i, 1) = *ptr++;
            }
        }


        template <typename scalar_type>
        void TSNE<scalar_type>::initializeClassEmbeddingPosition(int seed, double multiplier) {
            utils::secureLog(_logger, "Initializing the class glyph embedding...");
            if (seed < 0) {
                std::srand(static_cast<unsigned int>(time(NULL)));
            }
            else {
                std::srand(seed);
            }

            int num_class_last = _class_embedding->numDataPoints();
            int num_class_now = getclassnum();

            _class_embedding->resize(_init_params._embedding_dimensionality, num_class_now, 0);
            _class_embedding_container = &_class_embedding->getContainer();

            if (num_class_last == num_class_now) {
                std::cout << "number of classes does not change with " << num_class_now << "\n";
                for (int i = 0; i < num_class_now; ++i) {
                    std::cout << "x and y is" << _class_embedding->dataAt(i, 0) << "|" << _class_embedding->dataAt(i, 1) << "\n";
                }
                return;
            }
            else {
                std::cout << "previous number of classes is " << num_class_last << ", current number of classes is " << num_class_now << "\n";
                for (int i = num_class_last; i < num_class_now; ++i) {
                    double x(0.);
                    double y(0.);
                    double radius(0.);
                    do {
                        x = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
                        y = 2 * (rand() / ((double)RAND_MAX + 1)) - 1;
                        radius = (x * x) + (y * y);
                    } while ((radius >= 1.0) || (radius == 0.0));

                    radius = sqrt(-2 * log(radius) / radius);
                    x *= radius * multiplier;
                    y *= radius * multiplier;
                    std::cout << "x and y is" << x << "|" << y << "\n";
                    _class_embedding->dataAt(i, 0) = x;
                    _class_embedding->dataAt(i, 1) = y;
                }
            }
        }
        }
        }
#endif 
