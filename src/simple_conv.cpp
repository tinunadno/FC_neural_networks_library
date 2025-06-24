#include "simple_conv.h"
#include "sc_private.h"

namespace simple_conv {

    mkl_BLAS_impl::mat forward(const mkl_BLAS_impl::mat& input_layer, const net& net_) {
        using namespace mkl_BLAS_impl;

        mat processing_layer(input_layer);

        for(int i = 0; i < net_.size(); i+=2){
            mat tmp;
            gemm_y(&net_[i], &processing_layer, 1.f, 0, 0, &tmp);
            add_no_copy(&tmp, &net_[i + 1], 1.f);
            processing_layer = tmp;
            if (i + 2 >= net_.size()) {
                sc_private::apply_soft_max(processing_layer, processing_layer);
            } else {
                mkl_BLAS_impl::trash_hold(&processing_layer, &processing_layer);
            }
        }

        return processing_layer;
    }

    net generate_empty_net(const std::vector<int> &shapes) {
        using namespace mkl_BLAS_impl;
        net layers;
        layers.reserve((shapes.size() - 1) * 2);

        for (size_t i = 0; i < shapes.size() - 1; ++i) {
            int fan_in = shapes[i];
            int fan_out = shapes[i + 1];

            // Weights
            layers.emplace_back(fan_out, fan_in);
            cv::Mat weights(fan_out, fan_in, CV_32F, (*(layers.end() - 1)).data);
            cv::randu(weights, -.5f, .5f);

            // Biases
            layers.emplace_back(fan_out, 1);
            cv::Mat bias(fan_out, 1, CV_32F, (*(layers.end() - 1)).data);
            cv::randu(bias, -.5f, .5f);

        }

        return layers;
    }

} // simple_conv
