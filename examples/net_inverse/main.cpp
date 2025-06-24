#include "simple_conv.h"
#include "learning_private.h"

int main(){
    using namespace simple_conv;
    using namespace simple_conv::mkl_BLAS_impl;

    std::string base_path = CONV_HOME;
    std::string net_path = base_path + "data/net_.conv";

    auto net_ = io::read_net(net_path);

    int last_layer_size = (*(net_.end() - 1)).size();

    mat input(net_[0].cols, last_layer_size);
    cv::Mat temp_rnd(input.rows, input.cols, CV_32F, input.data);
    cv::randu(temp_rnd, -.5f, .5f);
    mat one_hot(last_layer_size, last_layer_size);
    mat labels(last_layer_size, 1);
    for(int i = 0; i < last_layer_size; i++){
        *(labels.data + i) = (float)i;
    }
    learning::learning_private::one_hot(labels, one_hot);

    auto hidden_layers = std::vector<mat>(net_.size());
    auto gradient = net(net_.size());

    mat input_grad(input.rows, input.cols);

    for (int i = 0; i < net_.size(); i += 2) {
        hidden_layers[i] = mat(net_[i + 1].rows, last_layer_size);
        hidden_layers[i + 1] = mat(net_[i + 1].rows, last_layer_size);
        gradient[i] = mat(net_[i].rows, net_[i].cols);
        gradient[i + 1] = mat(net_[i + 1].rows, net_[i + 1].cols);
    }

    float error = std::numeric_limits<float>::max();
    float threshold = 1e-5f;
    float grad_weight = .05f;

    auto mse = [](const mat& one_hot, const mat& result){
        float res = 0.f;
        float temp_res = .0f;
        CV_Assert(one_hot.add_possible(&result));
        for(int i = 0; i < one_hot.size(); i++){
            float tmp = *(one_hot.data + i) - *(result.data + i);
            res += tmp * tmp;
            if(std::isinf(res)){
                res = temp_res;
            }else{
                temp_res = res;
            }
        }
        return res / (float)one_hot.size();
    };

//    Mat dz1;
//    gemm(net[2], dz2, 1., cv::Mat(), 0, dz1, GEMM_1_T);
//    Mat relu_der;
//    compare(hidden_layers[0], cv::Scalar(0), relu_der, CMP_GT);
//    relu_der.convertTo(relu_der, CV_32F, 1./255);
//    multiply(dz1, relu_der, dz1);

    while(error > threshold){
        learning::learning_private::forward_propagation(input, net_, hidden_layers);
        mat delta;
        learning::learning_private::backward_propagation(hidden_layers, one_hot, input, gradient, net_, delta);
        gemm_y(&net_[0], &delta, 1., 0, 0, &input_grad, GEMM_T_1);
        add_no_copy(&input, &input_grad, -1.f * grad_weight);
        error = mse(one_hot, *(hidden_layers.end() - 1));
        std::cout << error << std::endl;
    }

    return 0;
}