#include <string>
#include <src/simple_conv.h>

//TODO add input size to binary file,
//TODO add ability to save alphabet to binary file

int main(){
    std::string base_path = CONV_HOME;
    std::string dataset_path = base_path + "data/train.csv";
    std::string write_back_path = base_path + "data/net_.conv";

    auto net = simple_conv::generate_empty_net({28 * 28, 16, 16, 10});

//    float W1[] = {-.5f, .5f, .5f, -.5f};
//    float B1[] = {-.3f, .4f};
//    float W2[] = {-.1f, .3f, .6f, -.7f};
//    float B2[] = {-.5f, .1f};

//    std::vector<simple_conv::mkl_BLAS_impl::mat> net = {
//            simple_conv::mkl_BLAS_impl::mat(cv::Mat(2, 2, CV_32F, W1)),
//            simple_conv::mkl_BLAS_impl::mat(cv::Mat(2, 1, CV_32F, B1)),
//            simple_conv::mkl_BLAS_impl::mat(cv::Mat(2, 2, CV_32F, W2)),
//            simple_conv::mkl_BLAS_impl::mat(cv::Mat(2, 1, CV_32F, B2))
//    };



    simple_conv::learning::apply_gradient_descend(net, dataset_path, true);
    simple_conv::io::save_net(net, write_back_path);
}