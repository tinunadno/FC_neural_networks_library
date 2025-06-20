
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <mkl.h>
#include "simple_conv.h"

namespace simple_conv {

    namespace mkl_BLAS_impl{
//        gemm(delta, ls.hidden_layers[i - 2], norm, cv::Mat(), 0, ls.gradient[i - 1], GEMM_2_T);

        void gemm_y(mat* src1, mat* src2, float alpha, mat* src3, float betta, mat* dest, int transpose_flags = GEMM_T_NO){
            auto t1 = transpose_flags & GEMM_T_1 ? CblasTrans : CblasNoTrans;
            auto t2 = transpose_flags & GEMM_T_2 ? CblasTrans : CblasNoTrans;

            CV_Assert(src1->is_valid() && src2->is_valid());
            CV_Assert(src1->mul_possible(src2, transpose_flags));

            if(src3 != nullptr && src3->is_valid() && betta != 0.){
                if(!dest->is_valid()){
                    *dest = mat(src3->rows, src3->cols);
                }
                CV_Assert(dest->rows == src3->rows && dest->cols == src3->cols);
                memcpy((char*)dest->data, (const char*)src3->data, src3->rows * src3->cols * sizeof(float));
            } else {
                if (!dest->is_valid()){
                    int r = transpose_flags & GEMM_T_1 ? src1->cols : src1->rows;
                    int c = transpose_flags & GEMM_T_2 ? src2->rows : src2->cols;
                    *dest = mat(r, c);
                }
                for(size_t i = 0; i < dest->cols * dest->rows; i++){
                    *(dest->data + i) = 0.;
                }
            }
            // if dest is initialized we gotta check its dimensions
            CV_Assert(!transpose_flags && dest->rows == src1->rows && dest->cols == src2->cols ||
                      transpose_flags & GEMM_T_1 && dest->rows == src1->cols && dest->cols == src2->cols ||
                      transpose_flags & GEMM_T_2 && dest->rows == src1->rows && dest->cols == src2->rows
                      );

            int m = transpose_flags & GEMM_T_1 ? src1->cols : src1->rows;
            int k = transpose_flags & GEMM_T_1 ? src1->cols : src1->rows;
            int n = transpose_flags & GEMM_T_2 ? src2->cols : dest->cols;

            cblas_sgemm(CblasRowMajor, t1, t2,
                        src1->rows, src2->rows, src1->cols, alpha,src1->data, src1->cols,
                        src2->data, src2->cols,
                        src3 == nullptr ? 1.f : betta, dest->data, dest->cols);
        }

        void add(mat* src1, mat* src2, float alpha, mat* dest){
            CV_Assert(src1->cols == src2->cols && src1->rows == src2->rows);
            if(dest->is_valid()){
                CV_Assert(src1->cols == dest->cols && src1->rows == dest->rows);
            }else{
                *dest = mat(src1->rows, src1->cols);
            }
            int total = src1->rows * src1->cols;
            cblas_saxpy(total, alpha, src2->data, 1, src1->data, 1);
        }

        void broadcast_column_vector(mat* src1, const mat* addition){
            CV_Assert(src1->rows == addition->rows);
            CV_Assert(addition->cols == 1);
            int cols = src1->cols;
            for(int col = 0; col < cols; col++){
                for(int row = 0; row < src1->rows; row++){
                    *(src1->data + col + row * cols) += *(addition->data + row);
                }
            }
        }
    }

    namespace sc_private {

        class profiler{
        public:
            profiler(): _measures(){}
            void start_measure(const std::string& label){
                _measures[label] = std::chrono::system_clock::now();
            }
            void stop_measure(const std::string& label){
                if(_measures.find(label) == _measures.end())
                    std::cout << "Unregistered label: " << label << std::endl;
                auto start = _measures[label];
                auto stop = std::chrono::system_clock::now();
                std::cout << label <<  ": " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() << std::endl;
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
                return; // избегаем деления на ноль или nan'ов
            }

            // Масштабируем входы по температуре
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
            mat w(fan_out, fan_in);
            cv::Mat weights(fan_out, fan_in, CV_32F, w.data);
            cv::randu(weights, -.5f, .5f);

            // Biases
            mat b(fan_out, 1);
            cv::Mat bias(fan_out, 1, CV_32F, b.data);
            cv::randu(bias, -.5f, .5f);

            layers.push_back({w, b});
        }

        return layers;
    }

    namespace learning {

        namespace learning_private {
            struct learning_resources{
                net& net;
                mkl_BLAS_impl::mat dev_labels, dev_inputs, train_labels, train_inputs;
                mkl_BLAS_impl::mat one_hot;
                std::vector<mkl_BLAS_impl::mat> hidden_layers, dev_hidden_layers, gradient;
                sc_private::profiler profiler_;
            };

            void apply_soft_max(cv::Mat& src, cv::Mat& dst){
                if (dst.empty()){
                    dst = cv::Mat(src.size(), src.type());
                }
                CV_Assert(src.cols == dst.cols);
                CV_Assert(src.rows == dst.rows);
                for(int col  = 0; col < src.cols; col++){

                    float max_val = -std::numeric_limits<float>::max();

                    for(int row = 0; row < src.rows; row++){
                        float val = src.at<float>(row, col);
                        if(val > max_val){
                            max_val = val;
                        }
                    }

                    float exp_sum = 0.f;
                    for(int row = 0; row < src.rows; row++){
                        float exp_val = sc_private::safe_exp(src.at<float>(row, col) - max_val);
                        exp_sum += exp_val;
                    }

                    if(exp_sum <= 1e-6 || std::isnan(exp_sum)){
                        float uniform = 1.f / static_cast<float>(src.rows);
                        for(int row = 0; row < src.rows; row++){
                            dst.at<float>(row, col) = uniform * src.at<float>(row, col);
                        }
                    }else{
                        for(int row = 0; row < src.rows; row++){
                            float exp_val = sc_private::safe_exp(src.at<float>(row, col) - max_val);
                            dst.at<float>(row, col) = exp_val / exp_sum;
                        }
                    }
                }
            }

            void broadcast_column_addition(cv::Mat& add_it_to_me, const cv::Mat& column_vec){
                for(int col = 0; col < add_it_to_me.cols; col++){
                    for(int row = 0; row < add_it_to_me.rows; row++){
                        add_it_to_me.at<float>(row, col) += column_vec.at<float>(row);
                    }
                }
            }

            void forward_propagation(const cv::Mat &input_layer, const std::vector<cv::Mat> &net,
                                     std::vector<cv::Mat> &hidden_layers) {
                using namespace cv;
                gemm(net[0], input_layer, 1, cv::Mat(), 0, hidden_layers[0]);
                broadcast_column_addition(hidden_layers[0], net[1]);
                threshold(hidden_layers[0], hidden_layers[1], 0.0, 0.0, THRESH_TOZERO);
                for(int i = 2; i < net.size(); i+=2){
                    gemm(net[i], hidden_layers[i - 1], 1, cv::Mat(), 0, hidden_layers[i]);
                    broadcast_column_addition(hidden_layers[i], net[i + 1]);
                    if(i + 2 >= net.size()){
                        learning_private::apply_soft_max(hidden_layers[i], hidden_layers[i + 1]);
                    }else {
                        threshold(hidden_layers[i], hidden_layers[i + 1], 0.0, 0.0, THRESH_TOZERO);
                    }
                }

            }

            void forward_propagation(learning_resources& ls) {
                using namespace cv;
                forward_propagation(ls.train_inputs, ls.net, ls.hidden_layers);
            }

            void one_hot(const cv::Mat& labels, cv::Mat& one_hot_mtx, int classes_count){
                one_hot_mtx = cv::Mat::zeros(classes_count, labels.cols, CV_32F);
                for(int i = 0; i < labels.cols; i++){
                    float label_idx = labels.at<float>(0, i);
                    one_hot_mtx.at<float>(static_cast<int>(label_idx), i) = 1.f;
                }
            }

            void backward_propagation(learning_resources& ls){
                using namespace cv;

                int hidden_layers_last = (int)ls.hidden_layers.size() - 1;
                ls.profiler_.start_measure("    delta0");
                Mat delta;
                add(ls.hidden_layers[hidden_layers_last], -ls.one_hot, delta);
                ls.profiler_.stop_measure("    delta0");
                float norm = 1.f/(float)ls.train_inputs.cols;

                for(int i = hidden_layers_last; i > 1; i-=2){
                    ls.profiler_.start_measure("    dWn");
                    gemm(delta, ls.hidden_layers[i - 2], norm, cv::Mat(), 0, ls.gradient[i - 1], GEMM_2_T);
                    ls.profiler_.stop_measure("    dWn");
                    ls.profiler_.start_measure("    dbn");
                    reduce(delta, ls.gradient[i], 1, REDUCE_SUM, CV_32F);
                    ls.gradient[i] *= norm;
                    ls.profiler_.stop_measure("    dbn");
                    ls.profiler_.start_measure("    delta_n");
                    gemm(ls.net[i -1], delta, 1, cv::Mat(), 0, delta, GEMM_1_T);
                    ls.profiler_.stop_measure("    delta_n");
                }
                ls.profiler_.start_measure("    dw0");
                mkl_BLAS_impl::mat train(ls.train_inputs.rows, ls.train_inputs.cols, (float*)ls.train_inputs.data);
                mkl_BLAS_impl::mat grad(ls.gradient[0].rows, ls.gradient[0].cols, (float*)ls.gradient[0].data);
                mkl_BLAS_impl::mat delta_(delta.rows, delta.cols, (float*)delta.data);
                mkl_BLAS_impl::gemm_y(&delta_, &train, norm, 0,0, &grad, mkl_BLAS_impl::transpose_flags::GEMM_T_2);
//                gemm(delta, ls.train_inputs, norm, cv::Mat(), 0, ls.gradient[0], GEMM_2_T);
                ls.profiler_.stop_measure("    dw0");
                ls.profiler_.start_measure("    db0");
                reduce(delta, ls.gradient[1], 1, REDUCE_SUM, CV_32F);
                ls.gradient[1] *= norm;
                ls.profiler_.stop_measure("    db0");
            }

            void update_params(learning_resources& ls, float grad_weight){
                for(int i = 0; i < (int)ls.net.size(); i++){
                    ls.net[i] -= grad_weight * ls.gradient[i];
                }
            }

            void get_predictions(const cv::Mat& last_layer, cv::Mat& predictions){
                predictions = cv::Mat(1, last_layer.cols, CV_32F);
                for(int i = 0; i < last_layer.cols; i++){
                    float max_val = last_layer.at<float>(0, i);
                    int best_idx = 0;
                    for(int j = 0; j < last_layer.rows; j++){
                        float val = last_layer.at<float>(j, i);
                        if(val > max_val){
                            max_val = val;
                            best_idx = j;
                        }
                    }
                    predictions.at<float>(0, i) = (float)best_idx;
                }
            }

            float get_accuracy(const cv::Mat& labels, const cv::Mat& predictions){
                int matches = 0;
                for(int i = 0; i < labels.cols; i++){
                    matches += abs(labels.at<float>(0, i) - predictions.at<float>(0, i)) < 1e-6;
                }
                return (float)matches / (float)labels.cols * 100;
            }

            //!!! YOU MUST MANUALLY FREE DATASET DATA
            cv::Mat load_dataset(const boost::filesystem::path &filename, char delimiter = ',', bool has_header = true) {

                if(!exists(filename)){
                    std::cerr << "Dataset path doesn't exists!" << std::endl;
                    throw std::exception();
                }
                int fd = open(filename.c_str(), O_RDONLY);
                if (fd == -1) {
                    throw std::runtime_error("Failed to open file: " + filename.string() + ", error: " + strerror(errno));
                }

                off_t size = lseek(fd, 0, SEEK_END);
                if (size <= 0) {
                    close(fd);
                    throw std::runtime_error("File is empty or invalid: " + filename.string());
                }

                void *data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
                close(fd);
                if (data == MAP_FAILED) {
                    throw std::runtime_error("Failed to mmap file: " + filename.string() + ", error: " + strerror(errno));
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
                        if(first_line){
                            num_cols++;
                        }
                        first_line = false;
                    }
                    ptr++;
                }

                size_t total = num_rows * num_cols;

                auto *mat_data = static_cast<float*>(malloc((total) * sizeof(float)));
                float *mat_data_ptr = mat_data;
                if(mat_data == nullptr){
                    munmap(data, size);
                    throw std::runtime_error("bad alloc!");
                }

                ptr = buffer;

                for(size_t i = 0; i < total; i++){
                    char* end_ptr;
                    float val = strtof(ptr, &end_ptr);
                    *(mat_data_ptr++) = val;
                    ptr = end_ptr + 1;
                }

                cv::Mat mat_((int)num_rows, (int)num_cols, CV_32F, mat_data);

                munmap(data, size);

                cv::transpose(mat_, mat_);

                return mat_;
            }
            learning_resources initialize_resources(std::vector<cv::Mat>& net, const boost::filesystem::path& dataset_path, int dev_size){
                using namespace mkl_BLAS_impl;
                using namespace std;
                using namespace cv;
                Mat data = learning_private::load_dataset(dataset_path);

                Mat dev_set = data(cv::Range::all(), cv::Range(0, dev_size));
                Mat train_set = data(cv::Range::all(), cv::Range(dev_size, data.cols));

                Mat dev_labels = dev_set.row(0);
                Mat dev_inputs = dev_set.rowRange(1, data.rows) / 255.f;

                Mat train_labels = train_set.row(0);
                Mat train_inputs = train_set.rowRange(1, data.rows) / 255.f;

                Mat one_hot;
                learning_private::one_hot(train_labels, one_hot, (net[net.size() - 1]).rows);

                vector<mat> hidden_layers(net.size());
                vector<mat> dev_hidden_layers(net.size());
                vector<mat> gradient(net.size());
                for(int i = 0; i < net.size() - 1; i+=2){
                    hidden_layers[i] = mat(net[i+1].rows, train_inputs.cols);
                    hidden_layers[i + 1] = mat(net[i+1].rows, train_inputs.cols);
                    dev_hidden_layers[i] = mat(net[i+1].rows, dev_inputs.cols);
                    dev_hidden_layers[i + 1] = mat(net[i+1].rows, dev_inputs.cols);
                    gradient[i] = mat(net[i].rows, net[i].cols);
                    gradient[i + 1] = mat(net[i + 1].rows, net[i + 1].cols);
                }

                return {net, dev_labels, dev_inputs, train_labels, train_inputs, one_hot, hidden_layers, dev_hidden_layers, gradient, sc_private::profiler()};
            }

            void show_accuracy_sample(const learning_resources& ls, int sample_size){
                using namespace cv;

                cv::Mat sample = ls.dev_inputs.colRange(cv::Range(0, sample_size));
                std::vector<Mat> hidden;
                learning_private::forward_propagation(sample, ls.net, hidden);
                cv::Mat predictions;
                get_predictions(hidden[hidden.size() - 1], predictions);

                int img_size = (int)sqrt(sample.rows);
                for(int i = 0; i < sample_size; i++){
                    Mat img = sample.col(i).clone();
                    img = img.reshape(1, {img_size, img_size});
                    compare(img, cv::Scalar(0), img, CMP_GT);
                    resize(img, img, Size(700, 700), INTER_LINEAR);
                    imshow("label: " + std::to_string(ls.dev_labels.at<float>(i)) + " prediction: "
                    + std::to_string(predictions.at<float>(i)), img);
                    waitKey(0);
                }
            }

        } // learning_private

        void apply_gradient_descend(std::vector<cv::Mat> &net, const boost::filesystem::path &dataset_path,
                                    bool show_progress, float grad_weight, int epoch_, int dev_size, int sample_size) {
            using namespace std;
            using namespace cv;

            auto ls = learning_private::initialize_resources(net, dataset_path, dev_size);

            for(int i = 0; i < epoch_; i++){
                ls.profiler_.start_measure("forward");
                learning_private::forward_propagation(ls);
                ls.profiler_.stop_measure("forward");
                ls.profiler_.start_measure("backward");
                learning_private::backward_propagation(ls);
                ls.profiler_.stop_measure("backward");
                learning_private::update_params(ls, grad_weight);
                if(i % 10 == 0 && show_progress){
                    learning_private::forward_propagation(ls.dev_inputs, net, ls.dev_hidden_layers);
                    cv::Mat predictions;
                    learning_private::get_predictions(ls.dev_hidden_layers[ls.dev_hidden_layers.size() - 1], predictions);
                    float accuracy = learning_private::get_accuracy(ls.dev_labels, predictions);
                    cout << "EPOCH: " << i << endl;
                    cout << "ACCURACY: " << accuracy << "%" << endl;
                }
            }
            if(sample_size > 0){
                learning_private::show_accuracy_sample(ls, sample_size);
            }
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

        void save_net(const std::vector<cv::Mat> &net, const boost::filesystem::path &path) {
            using namespace std;
            using namespace cv;

            int layer_count = static_cast<int>(net.size());
            size_t total_size = sizeof(int) + layer_count * 2 * sizeof(int);
            vector<Size> sizes(layer_count);
            for (int i = 0; i < layer_count; i++) {
                sizes[i] = net[i].size();
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

        std::vector<cv::Mat> read_net(const boost::filesystem::path &path) {
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
            vector<Mat> layers(layer_count);

            int w, h;
            for (int i = 0; i < layer_count; i++) {
                in.read(reinterpret_cast<char *>(&w), sizeof(int));
                in.read(reinterpret_cast<char *>(&h), sizeof(int));
                sizes[i] = Size(w, h);
            }
            for (int i = 0; i < layer_count; i++) {
                layers[i] = Mat(sizes[i], CV_32F);
                in.read(reinterpret_cast<char *>(layers[i].data), (long) (sizeof(float) * (long) layers[i].total()));
            }

            in.close();
            return layers;
        }
    } // io

} // simple_conv
