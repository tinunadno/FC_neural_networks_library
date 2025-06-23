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
            bool free_me = true;

            mat(int r, int c, bool zeros = false) {
                if(zeros){
                    data = (float *) calloc(r * c, sizeof(float));
                }else {
                    data = (float *) malloc(r * c * sizeof(float));
                }
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
                free_me = false;
                this->data = data;
            }

            mat(const cv::Mat& m){
                this->rows = m.rows;
                this->cols = m.cols;
                this->capacity = m.rows * m.cols;
                this->data = (float*)malloc(this->capacity * sizeof(float));
                if (!data) {
                    throw std::runtime_error("Bad alloc!");
                }
                memcpy((char*)this->data, m.data, capacity * sizeof(float));
            }

            mat() {
                data = nullptr;
                rows = 0.;
                cols = 0.;
                capacity = 0;
                free_me = false;
            }

//            ~mat() {
//                if(free_me)
//                    free(data);
//            }
//
            void manual_free(){
                free(data);
            }

            mat& operator=(mat m){
                swap(*this, m);
                return *this;
            }

            friend void swap(mat& a, mat& b) noexcept{
                using std::swap;
                swap(a.rows, b.rows);
                swap(a.cols, b.cols);
                swap(a.capacity, b.capacity);
                swap(a.data, b.data);
                swap(a.free_me, b.free_me);
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

            void fill(float val){
                for(int i = 0; i < this->cols*this->rows; i++){
                    *(this->data + i) = val;
                }
            }

            int size() const{
                return rows*cols;
            }

            [[nodiscard]] bool is_valid() const {
                return rows != 0 && cols != 0 && data != nullptr;
            }

            [[nodiscard]] bool size_eq(const mat *m) const {
                return m != nullptr && m->rows == rows && m->cols == cols && m->data != nullptr;
            }

            [[nodiscard]] bool mul_possible(const mat *m2, int tr_flags = GEMM_T_NO) const {
                return m2 != nullptr && (tr_flags == 0 && m2->rows == cols || tr_flags & GEMM_T_1 && m2->rows == rows || tr_flags & GEMM_T_2 && m2->cols == cols);
            }
            [[nodiscard]] bool add_possible(const mat *m2, int tr_flags = GEMM_T_NO) const {
                return m2 != nullptr && (tr_flags == 0 && m2->rows == rows && m2->cols == cols ||
                tr_flags && m2->rows == cols && m2->cols == rows);
            }

            float get(int row, int col) const{
                return *(this->data + row * cols + col);
            }

            void set(int row, int col, float value) const{
                *(this->data + row * cols + col) = value;
            }

        };
    }

    typedef std::vector<mkl_BLAS_impl::mat> net;

//    cv::Mat forward(const cv::Mat &input_layer, const std::vector<cv::Mat> &net_);

    net generate_empty_net(const std::vector<int> &shapes);

    namespace learning {
        void apply_gradient_descend(net &net, const boost::filesystem::path &dataset_path,
                                    bool show_progress = false,
                                    float grad_weight = .1f,
                                    int epoch_ = 500,
                                    int dev_size = 1000,
                                    int sample_size = -1);
    }

    namespace io {
        cv::Mat read_img_to_input_layer(const boost::filesystem::path &path);

        void save_net(const net &net, const boost::filesystem::path &path);

        net read_net(const boost::filesystem::path &path);
    }

} // simple_conv

#endif //CONV_LIB_LIBRARY_H
