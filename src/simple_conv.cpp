
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <mkl.h>
#include "simple_conv.h"

namespace simple_conv {

    namespace mkl_BLAS_impl {
//        gemm(delta, ls.hidden_layers[i - 2], norm, cv::Mat(), 0, ls.gradient[i - 1], GEMM_2_T);

        void gemm_y(const mat *src1, const mat *src2, float alpha, mat *src3, float betta, mat *dest,
                    int transpose_flags = GEMM_T_NO) {
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

        void add(mat *src1, mat *src2, float alpha, mat *dest) {
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

        void broadcast_column_vector(mat *src1, const mat *addition) {
            CV_Assert(src1->rows == addition->rows);
            CV_Assert(addition->cols == 1);
            int cols = src1->cols;
            for (int col = 0; col < cols; col++) {
                for (int row = 0; row < src1->rows; row++) {
                    *(src1->data + col + row * cols) += *(addition->data + row);
                }
            }
        }

        void trash_hold(const mat *src, mat *dst) {
            CV_Assert(src->rows == dst->rows && src->cols == dst->cols);
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

    namespace sc_private {

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

        inline float safe_exp(float x) {
            constexpr float MAX_EXP_ARG = 87.0f;
            constexpr float MIN_EXP_ARG = -100.0f;

            x = std::max(MIN_EXP_ARG, std::min(x, MAX_EXP_ARG));
            return std::exp(x);
        }

        inline float sigmoid(float x) {
            return 1 / (1 + safe_exp(-x));
        }

        inline float inverse_sigmoid(float sigmoid) {
            return sigmoid * (1 - sigmoid);
        }

        inline void apply_inverse_sigmoid_derivative(cv::Mat &layer) {
            for (int value_idx = 0; value_idx < layer.total(); value_idx++) {
                float *value_prt = (float *) layer.data + value_idx;
                *value_prt = inverse_sigmoid(*value_prt);
            }
        }

        inline void apply_soft_max(cv::Mat &processing_layer, float temperature = .5f) {
            auto *p_inp_data = (float *) processing_layer.data;
            const int num_values = (int) processing_layer.total();

            if (num_values == 0 || temperature <= 0.0f || std::isnan(temperature)) {
                return;
            }

            float max_val = -std::numeric_limits<float>::infinity();
            for (int i = 0; i < num_values; ++i) {
                float val = p_inp_data[i] / temperature;
                if (val > max_val) max_val = val;
            }

            float exp_sum = 0.0f;
            for (int i = 0; i < num_values; ++i) {
                p_inp_data[i] = safe_exp(p_inp_data[i] / temperature - max_val);
                exp_sum += p_inp_data[i];
            }

            if (exp_sum <= 0.0f || std::isnan(exp_sum)) {
                std::fill(p_inp_data, p_inp_data + num_values, 1.0f / (float) num_values);
                return;
            }

            for (int i = 0; i < num_values; ++i) {
                p_inp_data[i] /= exp_sum;
            }
        }

        inline void apply_sigmoid(cv::Mat &processing_layer) {
            auto *p_inp_data = (float *) processing_layer.data;
            for (size_t value_idx = 0; value_idx < processing_layer.total(); value_idx++) {
                *(p_inp_data + value_idx) = sigmoid(*(p_inp_data + value_idx));
            }
        }
    } // sc_private

    cv::Mat forward(const cv::Mat &input_layer, const std::vector<cv::Mat> &net) {
        using namespace cv;

        assert(input_layer.type() == CV_32F);

        Mat processing_layer = input_layer;

        for (size_t layer_idx = 0; layer_idx < net.size(); layer_idx += 2) {
            const auto &weights = net[layer_idx];
            const auto &bias = net[layer_idx + 1];

            assert(weights.type() == CV_32F);
            assert(bias.type() == CV_32F);

            // opencv will deal with bad Matrix shapes
            processing_layer = weights * processing_layer;
            processing_layer += bias;

            if (layer_idx + 2 >= net.size()) {
                sc_private::apply_soft_max(processing_layer);
            } else {
                sc_private::apply_sigmoid(processing_layer);
            }
        }

        return processing_layer;
    }

    net generate_empty_net(const std::vector<int> &shapes) {
        using namespace mkl_BLAS_impl;
        net layers;
        layers.reserve((shapes.size() - 1) * 2);

        for (size_t i = 0; i < shapes.size() - 1; ++i) {
            int fan_in = shapes[i];
            int fan_out = shapes[i + 1];

            // Weights
            layers.emplace_back(fan_out, fan_in);
            cv::Mat weights(fan_out, fan_in, CV_32F, (*(layers.end() - 1)).data);
            cv::randu(weights, -.5f, .5f);

            // Biases
            layers.emplace_back(fan_out, 1);
            cv::Mat bias(fan_out, 1, CV_32F, (*(layers.end() - 1)).data);
            cv::randu(bias, -.5f, .5f);

        }

        return layers;
    }

    namespace learning {

        namespace learning_private {
            struct perc_learning_resources {
                net &net_;
                mkl_BLAS_impl::mat data;                                                    // this thing should be properly freed
                mkl_BLAS_impl::mat dev_labels, dev_inputs, train_labels, train_inputs;
                mkl_BLAS_impl::mat one_hot;
                std::vector<mkl_BLAS_impl::mat> hidden_layers, dev_hidden_layers, gradient;
                sc_private::profiler profiler_;
            };

            void apply_relu_der(mkl_BLAS_impl::mat &dz, mkl_BLAS_impl::mat &z){
                CV_Assert(dz.cols == z.cols);
                CV_Assert(dz.rows == z.rows);
                int total = dz.rows * dz.cols;
                for(int i = 0; i < total; i++){
                    if(*(z.data + i) < 0){
                        *(dz.data + i) = 0;
                    }
                }
            }

            void apply_soft_max(mkl_BLAS_impl::mat &src, mkl_BLAS_impl::mat &dst) {
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

            void broadcast_column_addition(cv::Mat &add_it_to_me, const cv::Mat &column_vec) {
                for (int col = 0; col < add_it_to_me.cols; col++) {
                    for (int row = 0; row < add_it_to_me.rows; row++) {
                        add_it_to_me.at<float>(row, col) += column_vec.at<float>(row);
                    }
                }
            }

            void forward_propagation(const mkl_BLAS_impl::mat &input_layer, const net &net,
                                     std::vector<mkl_BLAS_impl::mat> &hidden_layers) {
                using namespace cv;
                mkl_BLAS_impl::gemm_y(&net[0], &input_layer, 1, 0, 0, &hidden_layers[0]);
                mkl_BLAS_impl::broadcast_column_vector(&hidden_layers[0], &net[1]);
                mkl_BLAS_impl::trash_hold(&hidden_layers[0], &hidden_layers[1]);
                for (int i = 2; i < net.size(); i += 2) {
                    mkl_BLAS_impl::gemm_y(&net[i], &hidden_layers[i - 1], 1, 0, 0, &hidden_layers[i]);
                    mkl_BLAS_impl::broadcast_column_vector(&hidden_layers[i], &net[i + 1]);
                    if (i + 2 >= net.size()) {
                        learning_private::apply_soft_max(hidden_layers[i], hidden_layers[i + 1]);
                    } else {
                        mkl_BLAS_impl::trash_hold(&hidden_layers[i], &hidden_layers[i + 1]);
                    }
                }
            }

            void forward_propagation(perc_learning_resources &ls) {
                using namespace cv;
                forward_propagation(ls.train_inputs, ls.net_, ls.hidden_layers);
            }

            void one_hot(const mkl_BLAS_impl::mat &labels, mkl_BLAS_impl::mat &one_hot_mtx) {
//                one_hot_mtx = mkl_BLAS_impl::mat(classes_count, labels.cols, true);
                for (int i = 0; i < labels.cols; i++) {
                    float label_idx = *(labels.data + i);
                    one_hot_mtx.set(static_cast<int>(label_idx), i, 1.f);
                }
            }

            void backward_propagation(perc_learning_resources &ls) {
                using namespace mkl_BLAS_impl;

                int hidden_layers_last = (int) ls.hidden_layers.size() - 1;
                mat delta;
                add(&ls.hidden_layers[hidden_layers_last], &ls.one_hot, -1.f, &delta);
                cv::Mat asd(delta.rows, delta.cols, CV_32F, delta.data);
                float norm = 1.f / (float) ls.train_inputs.cols;
                for (int i = hidden_layers_last; i > 1; i -= 2) {
                    gemm_y(&delta, &ls.hidden_layers[i - 2], norm, 0, 0, &ls.gradient[i - 1], GEMM_T_2);
                    reduce_columns(&delta, &ls.gradient[i]);
                    mul_scalar(&ls.gradient[i], norm);
                    mat nu_delta;
                    gemm_y(&ls.net_[i - 1], &delta, 1.f, 0, 0, &nu_delta, GEMM_T_1);
                    delta = nu_delta;
                    apply_relu_der(delta, ls.hidden_layers[i - 3]);
                }
                mkl_BLAS_impl::gemm_y(&delta, &ls.train_inputs, norm, 0, 0,
                                      &ls.gradient[0], mkl_BLAS_impl::transpose_flags::GEMM_T_2);
                reduce_columns(&delta, &ls.gradient[1]);
                mul_scalar(&ls.gradient[1], norm);

                cv::Mat asd0(ls.gradient[0].rows, ls.gradient[0].cols, CV_32F, ls.gradient[0].data);
                cv::Mat asd1(ls.gradient[1].rows, ls.gradient[1].cols, CV_32F, ls.gradient[1].data);
                cv::Mat asd2(ls.gradient[2].rows, ls.gradient[2].cols, CV_32F, ls.gradient[2].data);
                cv::Mat asd3(ls.gradient[3].rows, ls.gradient[3].cols, CV_32F, ls.gradient[3].data);
                int a = 0;
                (void)a;
            }

            void update_params(perc_learning_resources &ls, float grad_weight) {
                using namespace mkl_BLAS_impl;
                for (int i = 0; i < (int) ls.net_.size(); i++) {
                    add(&ls.net_[i], &ls.gradient[i], -grad_weight, &ls.net_[i]);
                }
            }

            void get_predictions(const mkl_BLAS_impl::mat &last_layer, mkl_BLAS_impl::mat &predictions) {
                predictions = mkl_BLAS_impl::mat(1, last_layer.cols);
                for (int i = 0; i < last_layer.cols; i++) {
                    float max_val = last_layer.get(0, i);
                    int best_idx = 0;
                    for (int j = 0; j < last_layer.rows; j++) {
                        float val = last_layer.get(j, i);
                        if (val > max_val) {
                            max_val = val;
                            best_idx = j;
                        }
                    }
                    predictions.set(0, i, (float) best_idx);
                }
            }

            float get_accuracy(const mkl_BLAS_impl::mat &labels, const mkl_BLAS_impl::mat &predictions) {
                int matches = 0;
                for (int i = 0; i < labels.cols; i++) {
                    matches += abs(labels.get(0, i) - predictions.get(0, i)) < 1e-6;
                }
                return (float) matches / (float) labels.cols * 100;
            }

            //!!! YOU MUST MANUALLY FREE DATASET DATA
            mkl_BLAS_impl::mat
            load_dataset(const boost::filesystem::path &filename, bool transposed = false, char delimiter = ',',
                         bool has_header = true) {

                if (!exists(filename)) {
                    std::cerr << "Dataset path doesn't exists!" << std::endl;
                    throw std::exception();
                }
                int fd = open(filename.c_str(), O_RDONLY);
                if (fd == -1) {
                    throw std::runtime_error(
                            "Failed to open file: " + filename.string() + ", error: " + strerror(errno));
                }

                off_t size = lseek(fd, 0, SEEK_END);
                if (size <= 0) {
                    close(fd);
                    throw std::runtime_error("File is empty or invalid: " + filename.string());
                }

                void *data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
                close(fd);
                if (data == MAP_FAILED) {
                    throw std::runtime_error(
                            "Failed to mmap file: " + filename.string() + ", error: " + strerror(errno));
                }

                const char *buffer = static_cast<const char *>(data);
                const char *end = buffer + size;

                if (has_header) {
                    const char *line_end = static_cast<const char *>(memchr(buffer, '\n', end - buffer));
                    if (!line_end) {
                        munmap(data, size);
                        throw std::runtime_error("No newline found in file.");
                    }
                    buffer = line_end + 1;
                }

                size_t num_rows = 0;
                size_t num_cols = 0;
                const char *ptr = buffer;
                bool first_line = true;

                while (ptr <= end) {
                    if (*ptr == delimiter) {
                        if (first_line) {
                            num_cols++;
                        }
                    } else if (*ptr == '\n') {
                        num_rows++;
                        if (first_line) {
                            num_cols++;
                        }
                        first_line = false;
                    }
                    ptr++;
                }

                size_t total = num_rows * num_cols;

                auto *mat_data = static_cast<float *>(malloc((total) * sizeof(float)));
                float *mat_data_ptr = mat_data;
                if (mat_data == nullptr) {
                    munmap(data, size);
                    throw std::runtime_error("bad alloc!");
                }

                ptr = buffer;
                size_t counter = 0;
                for (size_t i = 0; i < total; i++) {
                    char *end_ptr;
                    float val = strtof(ptr, &end_ptr);
                    if (transposed) {
                        int row = counter / num_cols;
                        int col = counter++ % num_cols;
                        *(mat_data_ptr + col * num_rows + row) = val;
                    } else {
                        *(mat_data_ptr++) = val;
                    }
                    ptr = end_ptr + 1;
                }

//                cv::Mat mat_((int)num_rows, (int)num_cols, CV_32F, mat_data);
//
                munmap(data, size);
//
//                cv::transpose(mat_, mat_);
                if(transposed) {
                    return mkl_BLAS_impl::mat((int) num_cols, (int) num_rows, (float *) mat_data);
                }else{
                    return mkl_BLAS_impl::mat((int) num_rows, (int) num_cols, (float *) mat_data);
                }
            }

            void
            initialize_resources(net &net_, const boost::filesystem::path &dataset_path, perc_learning_resources* ls, int dev_size) {
                using namespace mkl_BLAS_impl;
                using namespace std;
                using namespace cv;

                mat data = learning_private::load_dataset(dataset_path, true);

                Mat data_wrapper(data.rows, data.cols, CV_32F, data.data);

                Mat dev_set = data_wrapper(cv::Range::all(), cv::Range(0, dev_size));
                Mat train_set = data_wrapper(cv::Range::all(), cv::Range(dev_size, data_wrapper.cols));


                if(dev_size > 0) {
                    Mat dev_labels = dev_set.row(0);
                    Mat dev_inputs = dev_set.rowRange(1, data_wrapper.rows) / 255.f;
                    ls->dev_labels = mat(dev_labels);
                    ls->dev_inputs = mat(dev_inputs);
                }
                Mat train_labels = train_set.row(0);
                Mat train_inputs = train_set.rowRange(1, data_wrapper.rows) / 255.f;

                ls->one_hot = mat((net_[net_.size() - 1]).rows, train_labels.cols);
                learning_private::one_hot(train_labels, ls->one_hot);
                if(dev_size > 0) {
                    ls->dev_hidden_layers = vector<mat>(net_.size());
                }
                ls->hidden_layers = vector<mat>(net_.size());
                ls->gradient = net(net_.size());
                for (int i = 0; i < net_.size(); i += 2) {
                    ls->hidden_layers[i] = mat(net_[i + 1].rows, train_inputs.cols);
                    ls->hidden_layers[i + 1] = mat(net_[i + 1].rows, train_inputs.cols);
                    if(dev_size > 0) {
                        ls->dev_hidden_layers[i] = mat(net_[i + 1].rows, ls->dev_inputs.cols);
                        ls->dev_hidden_layers[i + 1] = mat(net_[i + 1].rows, ls->dev_inputs.cols);
                    }
                    ls->gradient[i] = mat(net_[i].rows, net_[i].cols);
                    ls->gradient[i + 1] = mat(net_[i + 1].rows, net_[i + 1].cols);
                }
                ls->train_labels = mat(train_labels);
                ls->train_inputs = mat(train_inputs);
                ls->profiler_ = sc_private::profiler();
            }

            void show_accuracy_sample(const perc_learning_resources &ls, int sample_size) { // TODO fix it
                using namespace cv;
                using namespace mkl_BLAS_impl;

                mat dev_sample(ls.dev_inputs.rows, sample_size, ls.dev_inputs.data);

                std::vector<mat> hidden;
                learning_private::forward_propagation(dev_sample, ls.net_, hidden);
                mat predictions;
                get_predictions(hidden[hidden.size() - 1], predictions);

                int img_size = (int) sqrt(dev_sample.rows);
                Mat sample_wrapper(dev_sample.rows, dev_sample.cols, CV_32F, dev_sample.data);
                for (int i = 0; i < sample_size; i++) {
                    Mat img = sample_wrapper.col(i).clone();
                    img = img.reshape(1, {img_size, img_size});
                    compare(img, cv::Scalar(0), img, CMP_GT);
                    resize(img, img, Size(700, 700), INTER_LINEAR);
                    imshow("label: " + std::to_string(ls.dev_labels.get(0, i)) + " prediction: "
                           + std::to_string(predictions.get(0, i)), img);
                    waitKey(0);
                }
            }

        } // learning_private

        void apply_gradient_descend(net &net, const boost::filesystem::path &dataset_path,
                                    bool show_progress, float grad_weight, int epoch_, int dev_size, int sample_size) {
            using namespace std;
            using namespace cv;
            learning_private::perc_learning_resources ls={net};
            learning_private::initialize_resources(net, dataset_path, &ls, dev_size);

            for (int i = 0; i < epoch_; i++) {
                learning_private::forward_propagation(ls);
                learning_private::backward_propagation(ls);
                learning_private::update_params(ls, grad_weight);
                if (i % 10 == 0 && show_progress && dev_size > 0) {
                    learning_private::forward_propagation(ls.dev_inputs, net, ls.dev_hidden_layers);
                    mkl_BLAS_impl::mat predictions;
                    learning_private::get_predictions(ls.dev_hidden_layers[ls.dev_hidden_layers.size() - 1],
                                                      predictions);
                    float accuracy = learning_private::get_accuracy(ls.dev_labels, predictions);
                    cout << "EPOCH: " << i << endl;
                    cout << "ACCURACY: " << accuracy << "%" << endl;
                }
            }
            if (sample_size > 0) {
                learning_private::show_accuracy_sample(ls, sample_size);
            }
            ls.data.manual_free();
        }
    } // learning

    namespace io {

        cv::Mat read_img_to_input_layer(const boost::filesystem::path &path) {
            using namespace cv;

            if (!exists(path)) {
                std::cerr << "image doesn't exists: " << path << std::endl;
                throw std::exception();
            }

            Mat img = imread(path.c_str(), CV_8U);
            img.convertTo(img, CV_32F, 1.0 / 255.0);
            img = 1.0 - img;

            img = img.reshape(1, (int) img.total());

            return img;
        }

        void save_net(const net &net, const boost::filesystem::path &path) {
            using namespace std;
            using namespace cv;

            int layer_count = static_cast<int>(net.size());
            size_t total_size = sizeof(int) + layer_count * 2 * sizeof(int);
            vector<Size> sizes(layer_count);
            for (int i = 0; i < layer_count; i++) {
                sizes[i] = Size(net[i].rows, net[i].cols);
                total_size += sizeof(float) * sizes[i].width * sizes[i].height;
            }

            size_t writing_pointer = sizeof(int);
            char *data = (char *) malloc(total_size);
            if (!data) {
                cerr << "bad alloc" << endl;
                throw exception();
            }
            (*(int *) data) = layer_count;

            for (const auto &i: sizes) {
                (*(int *) (data + writing_pointer)) = i.width;
                writing_pointer += sizeof(int);
                (*(int *) (data + writing_pointer)) = i.height;
                writing_pointer += sizeof(int);
            }


            for (int i = 0; i < net.size(); i++) {
                size_t current_size = sizes[i].width * sizes[i].height * sizeof(float);
                memcpy((char *) (data + writing_pointer), (char *) net[i].data, current_size);
                writing_pointer += current_size;
            }

            fstream out(path, ios::out | ios::binary);
            if (!out.is_open()) {
                cerr << "failed to open file" << endl;
                throw exception();
            }
            out.write(data, (long) total_size);
            out.close();

            free(data);
        }

        net read_net(const boost::filesystem::path &path) {
            using namespace std;
            using namespace cv;

            if (!exists(path)) {
                cerr << "this file doesn't exists: " << path.c_str() << endl;
                throw exception();
            }

            ifstream in(path, ios::binary);
            if (!in.is_open()) {
                cerr << "failed to open the file" << endl;
                throw exception();
            }

            int layer_count;
            in.read(reinterpret_cast<char *>(&layer_count), sizeof(int));

            vector<Size> sizes(layer_count);
            net layers(layer_count);

            int w, h;
            for (int i = 0; i < layer_count; i++) {
                in.read(reinterpret_cast<char *>(&w), sizeof(int));
                in.read(reinterpret_cast<char *>(&h), sizeof(int));
                sizes[i] = Size(w, h);
            }
            for (int i = 0; i < layer_count; i++) {
                layers[i] = mkl_BLAS_impl::mat(sizes[i].width, sizes[i].height);
                in.read(reinterpret_cast<char *>(layers[i].data), (long) (sizeof(float) * (long) layers[i].size()));
            }

            in.close();
            return layers;
        }
    } // io

} // simple_conv
