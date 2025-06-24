#ifndef CONV_LIB_GUI_EXAMPLE_LEARNING_PRIVATE_H
#define CONV_LIB_GUI_EXAMPLE_LEARNING_PRIVATE_H

#include "simple_conv.h"
#include "sc_private.h"

//! PS this thing is created to access some stuf inside lib, as in net_inverse example
namespace simple_conv::learning::learning_private {

    void apply_relu_der(mkl_BLAS_impl::mat &dz, const mkl_BLAS_impl::mat &z);

    void broadcast_column_addition(cv::Mat &add_it_to_me, const cv::Mat &column_vec);

    void forward_propagation(const mkl_BLAS_impl::mat &input_layer, const net &net,
                             std::vector<mkl_BLAS_impl::mat> &hidden_layers);


    void backward_propagation(const std::vector<mkl_BLAS_impl::mat>& hidden_layers,
                              const mkl_BLAS_impl::mat& one_hot,
                              const mkl_BLAS_impl::mat& train_inputs,
                              net& gradient,
                              const net& net_,
                              mkl_BLAS_impl::mat& delta); //! Ps left delta as an argument for outer access

    void one_hot(const mkl_BLAS_impl::mat &labels, mkl_BLAS_impl::mat &one_hot_mtx);

    void get_predictions(const mkl_BLAS_impl::mat &last_layer, mkl_BLAS_impl::mat &predictions);

    float get_accuracy(const mkl_BLAS_impl::mat &labels, const mkl_BLAS_impl::mat &predictions);

    float get_cross_entropy(const mkl_BLAS_impl::mat &labels, const mkl_BLAS_impl::mat &predictions);

} // learning_private

#endif //CONV_LIB_GUI_EXAMPLE_LEARNING_PRIVATE_H
