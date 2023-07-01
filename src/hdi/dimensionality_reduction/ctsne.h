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

#ifndef TSNE_H
#define TSNE_H

#include <vector>
#include <stdint.h>
#include <limits>
#include <string>
#include "../utils/assert_by_exception.h"
#include "../utils/abstract_log.h"
#include "../data/embedding.h"

namespace hdi {
    namespace dr {
        //! tSNE algorithm
        /*!
          Implementation of the tSNE algorithm
          \author Nicola Pezzotti
          \note
          This class is aimed at the analysis of very small data, therefore it lacks a lot of optimizations, e.g. reduntant data in P and Q
          It is written more for educational purposes than for a real usage
        */
        template <typename scalar_type = double>
        class TSNE {
        public:
            typedef std::vector<scalar_type> scalar_vector_type;
            typedef uint32_t data_handle_type;


        public:
            //! Parameters used for the initialization of the algorithm
            class InitParams {
            public:
                InitParams();
                double _perplexity;
                int _seed;
                int _embedding_dimensionality;

                double _minimum_gain_d;
                double _eta;
                double _eta_c;
                double _momentum;
                double _final_momentum;
                double _mom_switching_iter;
                double _exaggeration_factor;
                double _min_grad_norm;
            };


        public:
            TSNE();
            //! Get the dimensionality of the data
            void setDimensionality(int dimensionality) {
                checkAndThrowLogic(!_initialized, "Class should be uninitialized to change the dimensionality");
                checkAndThrowLogic(dimensionality > 0, "Invalid dimensionality");
                _dimensionality = dimensionality;
            }

            void setmetric(std::string metric) {
                _metric = metric;
            }

            //! Add a data-point to be processd
            //! \warning the class should not be initialized
            data_handle_type addDataPoint(const scalar_type* ptr);
            //! Initialize the class with the current data-points
            void initializeDataInfo(data::Embedding<scalar_type>* embedding);
            //! Reset the internal state of the class but it keeps the inserted data-points
            void reset();
            //! Reset the class and remove all the data points
            void clear();
            //! Return the number of data points
            int size() { return static_cast<int>(_high_dimensional_data.size()); }

            //! Get the dimensionality of the data
            int dimensionality() { return _dimensionality; }
            //! Get the high dimensional descriptor for a data point
            void getHighDimensionalDescriptor(scalar_vector_type& data_point, data_handle_type handle)const;
            //! Get the position in the embedding for a data point
            void getEmbeddingPosition(scalar_vector_type& embedding_position, data_handle_type handle)const;


            //! Get all the data points
            const std::vector<const scalar_type*>& getDataPoints()const { return _high_dimensional_data; }

            //! Get distances between data-points
            const scalar_vector_type& getDistancesSquared()const { return _distances_squared; }
            //! Get P
            const scalar_vector_type& getDistributionP()const { return _P_d; }
            //! Get Q
            const scalar_vector_type& getDistributionQ()const { return _Q_d; }
            //! Get Sigmas
            const scalar_vector_type& getSigmas()const { return _sigmas; }

            //! Return the current log
            utils::AbstractLog* logger()const { return _logger; }
            //! Set a pointer to an existing log
            void setLogger(utils::AbstractLog* logger) { _logger = logger; }

            //! Do an iteration of the gradient descent
            void doAnIteration(double alpha, double degrees_of_freedom, double lambda, double mult = 1);
            //! Compute the Kullback Leibler divergence
            double computeKullbackLeiblerDivergence();
            // Linhao: compute KL of class probability
            double computeKullbackLeiblerDivergenceClass();


            void setClassNum(int classnum) {
                _classnum = classnum;
            }
            //! Get the class number
            int getclassnum() { return _classnum; }
            //! Set probability
            void setProbabilityData(const scalar_type* ptr);
            //! Initialize class-glyph related info
            void initializeClassInfo(data::Embedding<scalar_type>* embedding);
            //! Initialize for gradient descent
            void initializeGradientDescent();

            void setClassEmbeddingPosition(const scalar_type* ptr);
            void setDataEmbeddingPosition(const scalar_type* ptr);

            void resetRemoveExaggerationIter(int iter) {
                _init_params._mom_switching_iter = iter;
            }

            void setExaggerationFactor(int factor) {
                _init_params._exaggeration_factor = factor;
            }

            void setPerplexity(int perplexity) {
                _init_params._perplexity = perplexity;
            }

            bool isconverged() {
                return isclassconverged && isdataconverged;
            }



        private:
            //! Compute the euclidean distances between points
            void computeHighDimensionalDistances();
            //! Compute a gaussian distribution for each data-point
            void computeGaussianDistributions(double perplexity);
            //! Compute High-dimensional distribution
            void compute_P_d();
            //! Initialize the point in the embedding
            void initializeEmbeddingPosition(int seed, double multipleir = .0001);
            //! Compute Low-dimensional distribution
            void compute_Q_d(double degrees_of_freedom);
            //! Compute tSNE gradient
            void computeGradient(double exaggeration, double alpha, double lambda, double degrees_of_freedom=1.);
            //! Update the embedding
            void updateTheEmbedding(double mult = 1.);

            //Compute high dimension distribution between class glyphs and data instances
            void compute_P_s();
            //! Initialize the point in the class embedding
            void initializeClassEmbeddingPosition(int seed, double multipleir = .0001);
            //Compute low dimension distribution between class glyphs and data instances
            void compute_Q_s(double degrees_of_freedom);



        private:
            double MACHINE_EPSILON = std::numeric_limits<float>::min();
            double MACHINE_MAX = std::numeric_limits<float>::max();
    
            bool isclassconverged;
            bool isdataconverged;

            int _dimensionality;
            int _classnum;
            std::vector<const scalar_type*> _high_dimensional_data; //! High-dimensional data
            data::Embedding<scalar_type>* _embedding; //! embedding
            typename data::Embedding<scalar_type>::scalar_vector_type* _embedding_container;

            scalar_vector_type _probability_data; //! Class probability data
            data::Embedding<scalar_type>* _class_embedding; //! class glyph embedding
            typename data::Embedding<scalar_type>::scalar_vector_type* _class_embedding_container;

            bool _initialized; //! Initialization flag

            scalar_vector_type _P_d; //! Conditional probalility distribution of data instances in the High-dimensional space
            scalar_vector_type _Q_d; //! Conditional probalility distribution of data instances in the Low-dimensional space
            scalar_type _normalization_Q_d; //! Normalization factor of Q - Z in the original paper

            scalar_vector_type _distances_squared; //! High-dimensional distances
            scalar_vector_type _sigmas; //! Sigmas of the gaussian probability distributions

            scalar_vector_type _P_s; //! class probalility in the High-dimensional space
            scalar_vector_type _Q_s; //! probalility distribution in the Low-dimensional space
            scalar_vector_type _normalization_Q_s; //! Normalization factor of Q - Z in the original paper

            // Gradient descent
            scalar_vector_type _gradient_d; //! Current gradient
            scalar_vector_type _previous_gradient_d; //! Previous gradient
            scalar_vector_type _gain_d; //! Gain

            scalar_vector_type _gradient_c; //! Current gradient
            scalar_vector_type _previous_gradient_c; //! Previous gradient
            scalar_vector_type _gain_c; //! Gain

            InitParams _init_params = InitParams();
            int _iteration;

            utils::AbstractLog* _logger;

            std::string _metric;

        };
    }
}
#endif 