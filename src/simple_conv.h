#ifndef CONV_LIB_LIBRARY_H
#define CONV_LIB_LIBRARY_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include "blas_impl.h"


namespace simple_conv {

    typedef std::vector<mkl_BLAS_impl::mat> net;

    mkl_BLAS_impl::mat forward(const mkl_BLAS_impl::mat& input_layer, const net& net_);

    net generate_empty_net(const std::vector<int> &shapes);

    namespace learning {
        void apply_gradient_descend(net &net, const boost::filesystem::path &dataset_path,
                                    bool show_progress = false,
                                    float grad_weight = .1f,
                                    int epoch_ = 1000,
                                    int dev_size = 1000,
                                    int sample_size = -1,
                                    int check_period = 10,
                                    int batch_size = 0,
                                    int patience = 15,
                                    float decay_factor = .7f);
    }

    namespace preprocessing{
        void preprocess_dataset(const boost::filesystem::path& in, const boost::filesystem::path& out);

        void crop_image(mkl_BLAS_impl::mat& img);

        void crop_image(cv::Mat& img, bool blur_me = false, int blur_size = 5);
    }

    namespace io {
        mkl_BLAS_impl::mat
        load_dataset(const boost::filesystem::path &filename, bool transposed = false,
                     char delimiter = ',', bool has_header = true);

        void save_dataset(const boost::filesystem::path &filename, mkl_BLAS_impl::mat& data, char delimiter = ',');

        cv::Mat read_img_to_input_layer(const boost::filesystem::path &path, bool invert = false, bool normalize = false);

        void save_net(const net &net, const boost::filesystem::path &path);

        net read_net(const boost::filesystem::path &path);
    }

} // simple_conv

#endif //CONV_LIB_LIBRARY_H
