#include <string>
#include <src/simple_conv.h>

//TODO add input size to binary file,
//TODO add ability to save alphabet to binary file

int main(){

    std::string base_path = CONV_HOME;
//!  Dataset cropping example
//    std::string dataset_path = base_path + "data/train.csv";
//    std::string processed_dataset_path = base_path + "data/processed_train.csv";
//    simple_conv::preprocessing::preprocess_dataset(dataset_path, processed_dataset_path);


    std::string dataset_path = base_path + "data/processed_train.csv";

    std::string write_back_path = base_path + "data/net_784_10_10.conv";

    auto net = simple_conv::generate_empty_net({28 * 28, 10, 10});



    simple_conv::learning::apply_gradient_descend(net, dataset_path, true);
    simple_conv::io::save_net(net, write_back_path);

    return 0;
}