#include "blas_impl.h"
#include <mkl.h>

namespace simple_conv::mkl_BLAS_impl {
//        gemm(delta, ls.hidden_layers[i - 2], norm, cv::Mat(), 0, ls.gradient[i - 1], GEMM_2_T);

    void gemm_y(const mat *src1, const mat *src2, float alpha, mat *src3, float betta, mat *dest,
                int transpose_flags) {
        auto t1 = transpose_flags & GEMM_T_1 ? CblasTrans : CblasNoTrans;
        auto t2 = transpose_flags & GEMM_T_2 ? CblasTrans : CblasNoTrans;

        CV_Assert(src1->is_valid() && src2->is_valid());
        CV_Assert(src1->mul_possible(src2, transpose_flags));

        if (src3 != nullptr && src3->is_valid() && betta != 0.) {
            if (!dest->is_valid()) {
                *dest = mat(src3->rows, src3->cols);
            }
            CV_Assert(dest->rows == src3->rows && dest->cols == src3->cols);
            memcpy((char *) dest->data, (const char *) src3->data, src3->rows * src3->cols * sizeof(float));
        } else {
            if (!dest->is_valid()) {
                int r = transpose_flags & GEMM_T_1 ? src1->cols : src1->rows;
                int c = transpose_flags & GEMM_T_2 ? src2->rows : src2->cols;
                *dest = mat(r, c);
            }
            for (size_t i = 0; i < dest->cols * dest->rows; i++) {
                *(dest->data + i) = 0.;
            }
        }
        // if dest is initialized we gotta check its dimensions
        CV_Assert(!transpose_flags && dest->rows == src1->rows && dest->cols == src2->cols ||
                  transpose_flags & GEMM_T_1 && dest->rows == src1->cols && dest->cols == src2->cols ||
                  transpose_flags & GEMM_T_2 && dest->rows == src1->rows && dest->cols == src2->rows
        );

        int m = transpose_flags & GEMM_T_1 ? src1->cols : src1->rows;
        int n = transpose_flags & GEMM_T_2 ? src2->rows : src2->cols;
        int k = transpose_flags & GEMM_T_1 ? src1->rows : src1->cols;

        cblas_sgemm(CblasRowMajor, t1, t2,
                    m, n, k, alpha, src1->data, src1->cols,
                    src2->data, src2->cols,
                    src3 == nullptr ? 1.f : betta, dest->data, dest->cols);
    }

    void add(const mat *src1, const mat *src2, float alpha, mat *dest) {
        CV_Assert(src1->cols == src2->cols && src1->rows == src2->rows);
        if (dest->is_valid()) {
            CV_Assert(src1->cols == dest->cols && src1->rows == dest->rows);
        } else {
            *dest = mat(src1->rows, src1->cols);
        }
        memcpy((char *) dest->data, (const char *) src1->data, src1->rows * src1->cols * sizeof(float));
        int total = src1->rows * src1->cols;
        cblas_saxpy(total, alpha, src2->data, 1, dest->data, 1);
    }

    void add_no_copy(mat *src1, const mat *src2, float alpha){
        CV_Assert(src1->cols == src2->cols && src1->rows == src2->rows);
        int total = src1->rows * src1->cols;
        cblas_saxpy(total, alpha, src2->data, 1, src1->data, 1);
    }

    void broadcast_column_vector(mat* src1, const mat* addition) {
        CV_Assert(src1->rows == addition->rows);
        int cols = src1->cols;
#pragma omp parallel for
        for (int col = 0; col < cols; col++) {
            cblas_saxpy(src1->rows, 1.0f,
                        addition->data, 1,
                        src1->data + col * src1->rows, 1);
        }
    }

    void apply_relu_der(mat &dz, const mat &z) {
        CV_Assert(dz.cols == z.cols);
        CV_Assert(dz.rows == z.rows);
        int total = dz.rows * dz.cols;
#pragma omp parralel for simd
        for (int i = 0; i < total; i++) {
            dz.data[i] = z.data[i] <= 0.f ? 0.f : dz.data[i];
        }
    }

    void trash_hold(const mat *src, mat *dst) {
        if(!dst->is_valid()){
            *dst = mat(src->rows, src->cols);
        }
        CV_Assert(src->rows == dst->rows && src->cols == dst->cols);
#pragma omp parralel for simd
        for (int i = 0; i < src->rows * src->cols; i++) {
            float val = *(src->data + i);
            *(dst->data + i) = val > 0.f ? val : 0.f;
        }
    }

    void reduce_columns(const mat *src, mat *dst) {
        CV_Assert(src->rows == dst->rows && dst->cols == 1);
        mat ones(src->cols, 1);
        ones.fill(1.f);
        gemm_y(src, &ones, 1.f, 0, 0, dst);
    }

    void mul_scalar(mat *m, float s) {
        cblas_sscal(m->rows * m->cols, s, m->data, 1);
    }

}