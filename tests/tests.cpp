#include <gtest/gtest.h>
#include <opencv4/opencv2/opencv.hpp>
#include "src/simple_conv.h"

TEST(conv_lib_test, io_test){
    using namespace std;
    using namespace cv;

    vector<Mat> layers = simple_conv::generate_empty_layers({28*28, 10, 10});

    // genering some data
    for(const auto& layer : layers){
        for(int i = 0; i < layer.total(); i++){
            *(float*)(layer.data + i) = (float)i;
        }
    }

    string basic_path = CONV_HOME;
    string path = basic_path + "/data/net.conv";

    simple_conv::io::save_net(layers, path);
    auto read_layers = simple_conv::io::read_net(path);

    ASSERT_EQ(layers.size(), read_layers.size());

    for(int i = 0; i < layers.size(); i++){
        int orig_layer_w = layers[i].size().width;
        int orig_layer_h = layers[i].size().height;
        int read_layer_w = read_layers[i].size().width;
        int read_layer_h = read_layers[i].size().height;

        ASSERT_EQ(orig_layer_h, read_layer_h);
        ASSERT_EQ(orig_layer_w, read_layer_w);
        ASSERT_EQ(layers[i].total(), read_layers[i].total());

        for(int j = 0; j < layers[i].total(); j++){
            ASSERT_EQ(*(float*)(layers[i].data + j), *(float*)(read_layers[i].data + j));
        }
    }
}

TEST(conv_lib_test, bad_read_path){
    std::string path = "/im_not_exist.conv";

    try{
        auto read_layers = simple_conv::io::read_net(path);
        // hehe
        ASSERT_EQ(1, 0);
    }catch(const std::exception& e){
        ASSERT_EQ(1, 1);
    }

}

TEST(conv_lib_test, read_img_test){
    using namespace std;
    string base = CONV_HOME;
    string path = base + "/data/test_img.png";

    cv::Mat img = simple_conv::io::read_img_to_input_layer(path);

    vector<cv::Mat> layers = simple_conv::generate_empty_layers({28*28, 10, 10});

    simple_conv::forward(img, layers);

}

TEST(conv_lib_test, bad_img_read_path){
    std::string path = "/im_not_exist.conv";

    try{
        auto read_layers = simple_conv::io::read_img_to_input_layer(path);
        // hehe
        ASSERT_EQ(1, 0);
    }catch(const std::exception& e){
        ASSERT_EQ(1, 1);
    }

}

TEST(conv_lib_test, mmap_load_test){
    std::string base = CONV_HOME;
    std::string path = base + "/data/train.csv";

    cv::Mat train_data = simple_conv::learning::learning_private::load_dataset(path);

    ASSERT_EQ(train_data.size[0], 42000);
    ASSERT_EQ(train_data.size[1], 785);

    free(train_data.data);
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}