#include <string>
#include <src/simple_conv.h>


int main(){
    std::string base_path = CONV_HOME;
    std::string dataset_path = base_path + "data/train.csv";
    std::string write_back_path = base_path + "data/net.conv";

    auto net = simple_conv::generate_empty_layers({28 * 28, 10, 10});

//    float W1[] = {-.5f, .5f, .5f, -.5f};
//    float B1[] = {-.3f, .4f};
//    float W2[] = {-.1f, .3f, .6f, -.7f};
//    float B2[] = {-.5f, .1f};
//
//    std::vector<cv::Mat> net = {
//            cv::Mat(2, 2, CV_32F, W1),
//            cv::Mat(2, 1, CV_32F, B1),
//            cv::Mat(2, 2, CV_32F, W2),
//            cv::Mat(2, 1, CV_32F, B2)
//    };



    simple_conv::learning::apply_gradient_descend(net, dataset_path, true);
    simple_conv::io::save_net(net, write_back_path);
}