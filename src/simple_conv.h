#ifndef CONV_LIB_LIBRARY_H
#define CONV_LIB_LIBRARY_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <boost/filesystem.hpp>

namespace simple_conv {

    cv::Mat forward(const cv::Mat &input_layer, const std::vector<cv::Mat> &net);

    std::vector<cv::Mat> generate_empty_layers(const std::vector<int> &shapes);

    namespace learning {
        void apply_gradient_descend(std::vector<cv::Mat> &net, const boost::filesystem::path &dataset_path,
                                    bool show_progress = false,
                                    int batch_size = 32, float trash_hold = .05f,
                                    float epsilon = 10e-5f, float grad_weight = .5f, int patience = 10,
                                    float decay_factor = .5f);
    }

    namespace io {
        cv::Mat read_img_to_input_layer(const boost::filesystem::path &path);

        void save_net(const std::vector<cv::Mat> &net, const boost::filesystem::path &path);

        std::vector<cv::Mat> read_net(const boost::filesystem::path &path);
    }

} // simple_conv

#endif //CONV_LIB_LIBRARY_H
