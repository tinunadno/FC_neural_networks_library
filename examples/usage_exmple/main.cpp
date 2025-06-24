#include <string>
#include <src/simple_conv.h>


int main() {
    std::string base_path = CONV_HOME;
    std::string net_path = base_path + "data/net_.conv";

    auto net = simple_conv::io::read_net(net_path);


    std::string img_path_1 = base_path + "data/1.png";
    std::string img_path_2 = base_path + "data/2.png";

    auto img_1 = simple_conv::io::read_img_to_input_layer(img_path_1, 1, 1);
    auto img_2 = simple_conv::io::read_img_to_input_layer(img_path_2, 1, 1);

    auto output_layer = simple_conv::forward(img_1, net);

    for (int i = 0; i < (int) output_layer.size(); i++) {
        std::cout << output_layer.get(0, i) << std::endl;
    }

    std::cout << std::endl;

    auto output_layer_1 = simple_conv::forward(img_2, net);

    for (int i = 0; i < (int) output_layer_1.size(); i++) {
        std::cout << output_layer_1.get(0, i) << std::endl;
    }

    return 0;
}