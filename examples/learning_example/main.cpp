#include <string>
#include <src/simple_conv.h>


int main(){
    std::string base_path = CONV_HOME;
    std::string dataset_path = base_path + "data/train.csv";
    std::string write_back_path = base_path + "data/net.conv";

    auto net = simple_conv::generate_empty_layers({28 * 28, 128, 10});

    simple_conv::learning::apply_gradient_descend(net, dataset_path, true);
    simple_conv::io::save_net(net, write_back_path);
}