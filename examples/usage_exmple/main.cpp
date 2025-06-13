#include <string>
#include <src/simple_conv.h>


int main() {
    std::string base_path = CONV_HOME;
    std::string net_path = base_path + "data/net.conv";
    std::string img_path = base_path + "data/1.png";

    auto net = simple_conv::io::read_net(net_path);
    auto img = simple_conv::io::read_img_to_input_layer(img_path);

    auto output_layer = simple_conv::forward(img, net);

    for (int i = 0; i < (int) output_layer.total(); i++) {
        std::cout << output_layer.at<float>(i) << std::endl;
    }
}