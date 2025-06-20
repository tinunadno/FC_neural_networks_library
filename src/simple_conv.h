#ifndef CONV_LIB_LIBRARY_H
#define CONV_LIB_LIBRARY_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <boost/filesystem.hpp>

namespace simple_conv {

    // TODO add 64bit alignment
    namespace mkl_BLAS_impl {
//        gemm(delta, ls.hidden_layers[i - 2], norm, cv::Mat(), 0, ls.gradient[i - 1], GEMM_2_T);
        enum transpose_flags {
            GEMM_T_NO = 0b0,
            GEMM_T_1 = 0b1,
            GEMM_T_2 = 0b10,
        };

        struct mat {
            float *data;
            int rows, cols;
            int capacity;

            mat(int r, int c) {
                data = (float *) malloc(r * c * sizeof(float));
                if (!data) {
                    throw std::runtime_error("Bad alloc!");
                }
                rows = r;
                cols = c;
                capacity = r * c;
            }

            mat(int r, int c, float *data) {
                capacity = r * c;
                rows = r;
                cols = c;
                this->data = data;
            }

            mat() {
                data = nullptr;
                rows = 0.;
                cols = 0.;
                capacity = 0;
            }

            ~mat() {
                free(data);
            }

            void reshape(int r, int c, bool force_realloc = false) {
                int new_size = r * c;
                int old_size = rows * cols;
                if (new_size < old_size / 2 || new_size > old_size || force_realloc) {
                    free(data);
                    data = (float *) malloc(new_size);
                    if (!data) {
                        throw std::runtime_error("Bad alloc!");
                    }
                }
                rows = r;
                cols = c;
            }

            [[nodiscard]] bool is_valid() const {
                return rows != 0 && cols != 0 && data != nullptr;
            }

            [[nodiscard]] bool size_eq(mat *m) const {
                return m != nullptr && m->rows == rows && m->cols == cols && m->data != nullptr;
            }

            [[nodiscard]] bool mul_possible(mat *m2, int tr_flags = GEMM_T_NO) const {
                return m2 != nullptr && (tr_flags == 0 && m2->rows == cols || tr_flags & GEMM_T_1 && m2->rows == rows || tr_flags & GEMM_T_2 && m2->cols == cols);
            }
            [[nodiscard]] bool add_possible(mat *m2, int tr_flags = GEMM_T_NO) const {
                return m2 != nullptr && (tr_flags == 0 && m2->rows == rows && m2->cols == cols ||
                tr_flags && m2->rows == cols && m2->cols == rows);
            }
        };
    }

    struct layer{
        mkl_BLAS_impl::mat w;
        mkl_BLAS_impl::mat b;
    };

    typedef std::vector<layer> net;

//    cv::Mat forward(const cv::Mat &input_layer, const std::vector<cv::Mat> &net);

    net generate_empty_net(const std::vector<int> &shapes);

    namespace learning {
        void apply_gradient_descend(std::vector<cv::Mat> &net, const boost::filesystem::path &dataset_path,
                                    bool show_progress = false,
                                    float grad_weight = .1f,
                                    int epoch_ = 1,
                                    int dev_size = 1000,
                                    int sample_size = -1);
    }

    namespace io {
        cv::Mat read_img_to_input_layer(const boost::filesystem::path &path);

        void save_net(const std::vector<cv::Mat> &net, const boost::filesystem::path &path);

        std::vector<cv::Mat> read_net(const boost::filesystem::path &path);
    }

} // simple_conv

#endif //CONV_LIB_LIBRARY_H
