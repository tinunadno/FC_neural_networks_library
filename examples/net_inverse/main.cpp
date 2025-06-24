#include "simple_conv.h"
#include "learning_private.h"

int main() {
    using namespace simple_conv;
    using namespace simple_conv::mkl_BLAS_impl;

    std::string base_path = CONV_HOME;
    std::string net_path = base_path + "data/net.conv";
    std::string wb_path = base_path + "data/perfect_one.png";

    auto net_ = io::read_net(net_path);

    int last_layer_size = (*(net_.end() - 1)).size();

    mat input(net_[0].cols, 1);
    cv::Mat temp_rnd(input.rows, input.cols, CV_32F, input.data);
    cv::randu(temp_rnd, -.5f, .5f);
    auto tmp = std::vector<float>({0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f});
    mat one_hot(last_layer_size, 1, tmp.data());

    auto hidden_layers = std::vector<mat>(net_.size());
    auto gradient = net(net_.size());

    mat input_grad(input.rows, input.cols);

    for (int i = 0; i < net_.size(); i += 2) {
        hidden_layers[i] = mat(net_[i + 1].rows, 1);
        hidden_layers[i + 1] = mat(net_[i + 1].rows, 1);
        gradient[i] = mat(net_[i].rows, net_[i].cols);
        gradient[i + 1] = mat(net_[i + 1].rows, net_[i + 1].cols);
    }

    float error = std::numeric_limits<float>::max();
    float threshold = 1e-6f;
    float grad_weight = .05f;

    auto mse = [](const mat &one_hot, const mat &result) {
        float res = 0.f;
        float temp_res = .0f;
        CV_Assert(one_hot.add_possible(&result));
        for (int i = 0; i < one_hot.size(); i++) {
            float tmp = *(one_hot.data + i) - *(result.data + i);
            res += tmp * tmp;
            if (std::isinf(res)) {
                res = temp_res;
            } else {
                temp_res = res;
            }
        }
        return res / (float) one_hot.size();
    };

    while (error > threshold) {
        learning::learning_private::forward_propagation(input, net_, hidden_layers);
        mat delta;
        learning::learning_private::backward_propagation(hidden_layers, one_hot, input, gradient, net_, delta);
        gemm_y(&net_[0], &delta, 1., 0, 0, &input_grad, GEMM_T_1);
        add_no_copy(&input, &input_grad, -1.f * grad_weight);
        error = mse(one_hot, *(hidden_layers.end() - 1));
    }

    auto out = forward(input, net_);
    for (int i = 0; i < out.size(); i++) {
        std::cout << out.get(0, i) << std::endl;
    }

    cv::Mat img(28, 28, CV_32F, input.data);
    img.convertTo(img, CV_8UC1, 255.f);
    cv::imwrite(wb_path, img);

    return 0;
}