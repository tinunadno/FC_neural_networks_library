
#include <fstream>
#include "simple_conv.h"

namespace simple_conv {

    namespace sc_private {
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

        inline void apply_soft_max(cv::Mat &processing_layer) {
            auto *p_inp_data = (float *) processing_layer.data;
            float exp_sum = 0;
            for (int value_idx = 0; value_idx < processing_layer.total(); value_idx++) {
                float *current_cell = p_inp_data + value_idx;
                float exp_ = safe_exp(*current_cell);
                exp_sum += exp_;
                *current_cell = exp_;
            }

            for (int value_idx = 0; value_idx < processing_layer.total(); value_idx++) {
                float *current_cell = p_inp_data + value_idx;
                *(current_cell) = *(current_cell) / exp_sum;
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

    std::vector<cv::Mat> generate_empty_layers(const std::vector<int> &shapes) {
        using namespace std;
        using namespace cv;

        vector<Mat> layers((shapes.size() - 1) * 2);

        for (int i = 0; i < shapes.size() - 1; i++) {
            layers[i * 2] = Mat(shapes[i + 1], shapes[i], CV_32F);
            layers[i * 2 + 1] = Mat(shapes[i + 1], 1, CV_32F);
        }

        return layers;
    }

    namespace learning {

        namespace learning_private {

            // PS I've repeated here, cuz I don't want to overcomplicate basic forward_with_layers interface
            void forward_saving_layers(const cv::Mat &input_layer, const std::vector<cv::Mat> &net,
                                       std::vector<cv::Mat> &hidden_layers) {
                using namespace cv;

                assert(input_layer.type() == CV_32F);

                Mat processing_layer = input_layer;

                hidden_layers.push_back(processing_layer);

                for (size_t layer_idx = 0; layer_idx < net.size(); layer_idx += 2) {
                    const auto &weights = net[layer_idx];
                    const auto &bias = net[layer_idx + 1];

                    assert(weights.type() == CV_32F);
                    assert(bias.type() == CV_32F);

                    // opencv will deal with bad Matrix shapes
                    processing_layer = weights * processing_layer;
                    processing_layer += bias;
                    hidden_layers.push_back(processing_layer.clone());
                    if (layer_idx + 2 >= net.size()) {
                        sc_private::apply_soft_max(processing_layer);
                    } else {
                        sc_private::apply_sigmoid(processing_layer);
                    }
                    hidden_layers.push_back(processing_layer.clone());
                }
            }

            float compute_loss_MSE(const cv::Mat &output, const cv::Mat &expected) {
                cv::Mat diff = output - expected;
                cv::Mat squared;
                cv::pow(diff, 2, squared);
                return static_cast<float>(cv::sum(squared)[0]) / (float)output.total();
            }


            void apply_back_prob_for_gradient(
                    const std::vector<cv::Mat> &net,
                    const cv::Mat &expected,
                    const std::vector<cv::Mat> &hidden_layers,
                    std::vector<cv::Mat> &gradient,
                    int batch_size)
            {
                using namespace cv;

                // hidden layer format is [a0, z1, a1, z2, a2,...]
                int num_layers = net.size() / 2; // number of (W,b) pairs

                // Start with output layer
                Mat aL = hidden_layers.back(); // activation of last layer
                Mat zL = hidden_layers[hidden_layers.size() - 2].clone(); // pre-activation of last layer

                // Output error
                Mat delta = (aL - expected); // dC/da^L
                sc_private::apply_inverse_sigmoid_derivative(zL);
                delta = delta.mul(zL); // dC/dz^L

                for (int l = num_layers - 1; l >= 0; --l) {
                    int weight_idx = l * 2;
                    int bias_idx = weight_idx + 1;

                    // Get activation from previous layer
                    int a_prev_idx = l * 2;
                    Mat a_prev = hidden_layers[a_prev_idx];
                    if (a_prev.cols > 1) { // if it's row vector, transpose
                        a_prev = a_prev.t();
                    }

                    // Compute gradients
                    Mat dW = delta * a_prev.t(); // dC/dW^l
                    Mat db = delta.clone(); // dC/db^l

                    // Normalize by batch size
                    dW /= batch_size;
                    db /= batch_size;

                    // Accumulate gradients
                    gradient[weight_idx] += dW.t();
                    gradient[bias_idx] += db.t();

                    // Propagate error backward if not input layer
                    if (l > 0) {
                        Mat W = net[weight_idx];
                        delta = W.t() * delta; // Error for previous layer

                        // Get z from previous layer
                        Mat z_prev = hidden_layers[a_prev_idx - 1].clone();
                        sc_private::apply_inverse_sigmoid_derivative(z_prev);

                        delta = delta.mul(z_prev); // updating delta
                    }
                }
            }

            void calculate_gradient(const std::vector<cv::Mat> &net, std::vector<cv::Mat> &gradient,
                                    const std::vector<cv::Mat> &input_batch,
                                    const std::vector<cv::Mat> &input_batch_expected,
                                    float& avg_error) {
                int batch_size = (int) input_batch.size();
                for (int input_idx = 0; input_idx < input_batch.size(); input_idx++) {
                    std::vector<cv::Mat> hidden_layers;
                    forward_saving_layers(input_batch[input_idx], net, hidden_layers);
                    avg_error += compute_loss_MSE(input_batch_expected[input_idx],
                                                  hidden_layers[hidden_layers.size() - 1]);
                    apply_back_prob_for_gradient(net, input_batch_expected[input_idx], hidden_layers,
                                                 gradient, batch_size);
                }
                avg_error /= (float) batch_size;
            }

            //! this thing is for csv only!
            void load_dataset(const boost::filesystem::path& path, std::vector<cv::Mat>& inputs, std::vector<int>& labels) {
                using namespace std;

                if(!exists(path)){
                    cerr << "File doesn't exists: " << path.c_str() << endl;
                    throw exception();
                }

                ifstream file(path);
                if (!file.is_open()) {
                    cerr << "Failed to open file: " << path << endl;
                    throw exception();
                }

                string line;
                bool first_line = true;

                while (getline(file, line)) {
                    if (first_line) {
                        first_line = false;
                        continue;
                    }

                    stringstream ss(line);
                    string item;
                    vector<uint8_t> row;

                    while (getline(ss, item, ',')) {
                        row.push_back(stoi(item));
                    }

                    int label = static_cast<int>(row[0]);
                    cv::Mat img((int)row.size() - 1, 1, CV_8U, row.data() + 1);
                    img.convertTo(img, CV_32F, 1.0 / 255.0);

                    labels.push_back(label);
                    inputs.push_back(img.clone());
                }
                file.close();

                cout << "Loaded " << inputs.size() << " inputs" << endl;
            }

        } // learning_private

        void apply_gradient_descend(std::vector<cv::Mat> &net, const boost::filesystem::path &dataset_path,bool show_progress, int batch_size, float trash_hold,
                                    float epsilon, float grad_weight, int patience, float decay_factor) {
            using namespace std;
            using namespace cv;

            vector<Mat> inputs;
            vector<int> labels;

            learning_private::load_dataset(dataset_path, inputs, labels);

            vector<Mat> current_batch(batch_size);
            vector<Mat> current_expected(batch_size);
            vector<Mat> gradient(net.size());
            for(int i = 0; i < gradient.size(); i++){
                gradient[i] = Mat(net[i].size[1], net[i].size[0], CV_32F);
            }
            float best_error = numeric_limits<float>::max();
            int stagnation_count = 0;

            size_t inp_size = inputs.size();
            for(size_t batch_index = 0; batch_index < inp_size; batch_index += batch_size){
                int current_batch_size = (int)(batch_size < inp_size - batch_index ? batch_size: inp_size - batch_index);
                for(int i = 0; i < current_batch_size; i++){
                    current_batch[i] = inputs[batch_index + i];
                    int current_label = labels[batch_index + i];
                    current_expected[i] = Mat((int)net[net.size() - 1].total(), 1, CV_32F);
                    current_expected[i].at<float>(current_label) = 1.f;
                }
                for(auto& grad_layer : gradient){
                    grad_layer = Mat::zeros(grad_layer.size(), grad_layer.type());
                }
                float current_error;
                learning_private::calculate_gradient(net, gradient, current_batch,
                                                     current_expected, current_error);
                if (current_error < trash_hold){
                    break;
                }
                if(current_error < best_error){
                    stagnation_count = 0;
                    best_error = current_error;
                }else{
                    stagnation_count++;
                    if(stagnation_count > patience){
                        grad_weight *= decay_factor;
                        stagnation_count = 0;
                    }
                }
                for(int i = 0; i < net.size(); i++){
                    net[i] -= gradient[i].t() * grad_weight;
                }

                if (show_progress){
                    cout << "current batch: " << batch_index << "/" << inp_size << " current error: " << current_error << endl;
                }
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

            Mat img = imread(path.c_str());
            if (img.type() == CV_8UC3) {
                cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
                img.convertTo(img, CV_32F, 1.0 / 255.0);
            } else {
                std::cerr << "got unsupportable image type (U8 gray or rgba)" << std::endl;
                throw std::exception();
            }

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
