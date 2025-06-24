#include <src/simple_conv.h>

int main() {

    std::string base_path = CONV_HOME;
    std::string net_path = base_path + "data/net_.conv";

    std::string write_back_path = base_path + "data/weights_as_images/";

    auto net = simple_conv::io::read_net(net_path);

    for (int l = 0; l < 4; l+=2) {
        const auto &weight = net[l]; // visualizing only first two layers weights, because they can be interpreted as an images (eg converted to a square shape)
        int size = (int) sqrtf((float) weight.cols);
        for (int n_idx = 0; n_idx < weight.rows; n_idx++) {
            cv::Mat neuron_weights = cv::Mat(1, weight.cols, CV_32F, weight.data + weight.cols * n_idx);
            neuron_weights.convertTo(neuron_weights, CV_8UC1, 255.0);
            neuron_weights = neuron_weights.reshape(1, {size, size});
            std::string img_name =
                    write_back_path + "layer" + std::to_string(l) + "neuron_" + std::to_string(n_idx) + "weights.png";
            cv::imwrite(img_name, neuron_weights);
        }
    }

    return 0;
}