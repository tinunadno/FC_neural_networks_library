#ifndef CONV_LIB_LEARNING_EXAMPLE_SC_PRIVATE_H
#define CONV_LIB_LEARNING_EXAMPLE_SC_PRIVATE_H

#include "blas_impl.h"
#include <unordered_map>
#include <chrono>
#include <iostream>

namespace simple_conv::sc_private {

    class profiler {
    public:
        profiler() : _measures() {}

        void start_measure(const std::string &label) {
            _measures[label] = std::chrono::system_clock::now();
        }

        void stop_measure(const std::string &label) {
            if (_measures.find(label) == _measures.end())
                std::cout << "Unregistered label: " << label << std::endl;
            auto start = _measures[label];
            auto stop = std::chrono::system_clock::now();
            std::cout << label << ": "
                      << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
            _measures.erase(label);
        }

    private:
        std::unordered_map<std::string, std::chrono::time_point<std::chrono::system_clock>> _measures;
    };

    inline float safe_exp(float x);

    void apply_soft_max(const mkl_BLAS_impl::mat &src, mkl_BLAS_impl::mat &dst);
} // sc_private
#endif //CONV_LIB_LEARNING_EXAMPLE_SC_PRIVATE_H
