#include "sc_private.h"

namespace simple_conv::sc_private {
    inline float safe_exp(float x) {
        constexpr float MAX_EXP_ARG = 87.0f;
        constexpr float MIN_EXP_ARG = -100.0f;

        x = std::max(MIN_EXP_ARG, std::min(x, MAX_EXP_ARG));
        return std::exp(x);
    }

    void apply_soft_max(const mkl_BLAS_impl::mat &src, mkl_BLAS_impl::mat &dst) {
        using namespace mkl_BLAS_impl;
        if (!dst.is_valid()) {
            dst = mat(src.rows, src.cols);
        }
        CV_Assert(src.cols == dst.cols);
        CV_Assert(src.rows == dst.rows);
        for (int col = 0; col < src.cols; col++) {

            float max_val = -std::numeric_limits<float>::max();

            for (int row = 0; row < src.rows; row++) {
                float val = src.get(row, col);
                if (val > max_val) {
                    max_val = val;
                }
            }

            float exp_sum = 0.f;
            for (int row = 0; row < src.rows; row++) {
                float exp_val = sc_private::safe_exp(src.get(row, col) - max_val);
                exp_sum += exp_val;
            }

            if (exp_sum <= 1e-6 || std::isnan(exp_sum)) {
                float uniform = 1.f / static_cast<float>(src.rows);
                for (int row = 0; row < src.rows; row++) {
                    dst.set(row, col, uniform * src.get(row, col));
                }
            } else {
                for (int row = 0; row < src.rows; row++) {
                    float exp_val = sc_private::safe_exp(src.get(row, col) - max_val);
                    dst.set(row, col, exp_val / exp_sum);
                }
            }
        }
    }
}