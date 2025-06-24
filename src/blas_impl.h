#ifndef CONV_LIB_LEARNING_EXAMPLE_BLAS_IMPL_H
#define CONV_LIB_LEARNING_EXAMPLE_BLAS_IMPL_H

#include <stdexcept>
#include <cstring>
#include <opencv2/core/mat.hpp>


namespace simple_conv::mkl_BLAS_impl {
//        gemm(delta, ls.hidden_layers[i - 2], norm, cv::Mat(), 0, ls.gradient[i - 1], GEMM_2_T);
    enum transpose_flags {
        GEMM_T_NO = 0b0,
        GEMM_T_1 = 0b1,
        GEMM_T_2 = 0b10,
    };

    struct roi {
        int xul, yul, xlr, ylr;

        roi(int x1, int y1, int x2, int y2) {
            xul = std::min(x1, x2);
            xlr = std::max(x1, x2);
            yul = std::min(y1, y2);
            ylr = std::max(y1, y2);
        }
    };

    struct mat {
        float *data;
        int rows, cols;
        int capacity;
        bool free_me = true;

        mat(int r, int c, bool zeros = false) {
            if (zeros) {
                data = (float *) calloc(r * c, sizeof(float));
            } else {
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

        mat(const mat &m) {
            rows = m.rows;
            cols = m.cols;
            capacity = rows * cols;
            data = (float *) malloc(rows * cols * sizeof(float));
            if (!data) {
                throw std::runtime_error("Bad alloc!");
            }
            memcpy((char *) this->data, m.data, capacity * sizeof(float));
        }

        mat(const cv::Mat &m) {
            this->rows = m.rows;
            this->cols = m.cols;
            this->capacity = m.rows * m.cols;
            this->data = (float *) malloc(this->capacity * sizeof(float));
            if (!data) {
                throw std::runtime_error("Bad alloc!");
            }
            memcpy((char *) this->data, m.data, capacity * sizeof(float));
        }

        mat(const mat &m, roi roi_) {
            cols = roi_.xlr - roi_.xul;
            rows = roi_.ylr - roi_.yul;
            capacity = rows * cols;
            this->data = (float *) malloc(this->capacity * sizeof(float));
            if (!data) {
                throw std::runtime_error("Bad alloc!");
            }
            for (int col = roi_.xul; col < roi_.xlr; col++) {
                for (int row = roi_.yul; row < roi_.ylr; row++) {
                    this->set(row - roi_.xul, col - roi_.yul, m.get(row, col));
                }
            }
        }

        mat() {
            data = nullptr;
            rows = 0.;
            cols = 0.;
            capacity = 0;
            free_me = false;
        }

        ~mat() {
            if (free_me)
                free(data);
        }

        void manual_free() {
            free(data);
        }

        mat &operator=(mat m) {
            swap(*this, m);
            return *this;
        }

        friend void swap(mat &a, mat &b) noexcept {
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

        void fill(float val) {
            for (int i = 0; i < this->cols * this->rows; i++) {
                *(this->data + i) = val;
            }
        }

        [[nodiscard]] int size() const {
            return rows * cols;
        }

        [[nodiscard]] bool is_valid() const {
            return rows != 0 && cols != 0 && data != nullptr;
        }

        [[nodiscard]] bool size_eq(const mat *m) const {
            return m != nullptr && m->rows == rows && m->cols == cols && m->data != nullptr;
        }

        [[nodiscard]] bool mul_possible(const mat *m2, int tr_flags = GEMM_T_NO) const {
            return m2 != nullptr && (tr_flags == 0 && m2->rows == cols || tr_flags & GEMM_T_1 && m2->rows == rows ||
                                     tr_flags & GEMM_T_2 && m2->cols == cols);
        }

        [[nodiscard]] bool add_possible(const mat *m2, int tr_flags = GEMM_T_NO) const {
            return m2 != nullptr && (tr_flags == 0 && m2->rows == rows && m2->cols == cols ||
                                     tr_flags && m2->rows == cols && m2->cols == rows);
        }

        float get(int row, int col) const {
            return *(this->data + row * cols + col);
        }

        void set(int row, int col, float value) const {
            *(this->data + row * cols + col) = value;
        }

    };

    void gemm_y(const mat *src1, const mat *src2, float alpha, mat *src3, float betta, mat *dest,
                int transpose_flags = GEMM_T_NO);

    void add(const mat *src1, const mat *src2, float alpha, mat *dest);

    void add_no_copy(mat *src1, const mat *src2, float alpha);

    void broadcast_column_vector(mat *src1, const mat *addition);

    void trash_hold(const mat *src, mat *dst);

    void reduce_columns(const mat *src, mat *dst);

    void mul_scalar(mat *m, float s);
}

#endif //CONV_LIB_LEARNING_EXAMPLE_BLAS_IMPL_H
