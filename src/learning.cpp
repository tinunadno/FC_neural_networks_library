#include "simple_conv.h"
#include "sc_private.h"

namespace simple_conv::learning {

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
                    sc_private::apply_soft_max(hidden_layers[i], hidden_layers[i + 1]);
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

        float get_cross_entropy(const mkl_BLAS_impl::mat &labels, const mkl_BLAS_impl::mat &predictions){
            float loss = 0.f;
            for(int i = 0; i < predictions.size(); i++){
                float clipped_pred = std::max(1e-7f, std::min(1.f - 1e-7f, predictions.get(0, i)));
                loss -= labels.get(0, i) * std::log(clipped_pred);
            }
            return loss;
        }

        void
        initialize_resources(net &net_, const boost::filesystem::path &dataset_path, perc_learning_resources* ls, int dev_size) {
            using namespace mkl_BLAS_impl;
            using namespace std;
            using namespace cv;

            mat data = io::load_dataset(dataset_path, true);

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

            mat dev_sample(ls.dev_inputs, mkl_BLAS_impl::roi(0, 0, sample_size, ls.dev_inputs.rows));

            std::vector<mat> hidden(ls.hidden_layers.size());
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


//        bool show_progress = false,
//        float grad_weight = .1f,
//        int epoch_ = 1000,
//        int dev_size = 1000,
//        int sample_size = 4);
    void apply_gradient_descend(net &net, const boost::filesystem::path &dataset_path,
                                bool show_progress,
                                float grad_weight,
                                int epoch_,
                                int dev_size,
                                int sample_size,
                                int check_period,
                                int batch_size,
                                int patience,
                                float decay_factor) {
        using namespace std;
        using namespace cv;

        learning_private::perc_learning_resources ls={net};
        learning_private::initialize_resources(net, dataset_path, &ls, dev_size);

        float best_error = numeric_limits<float>::max();
        int stagnation_count = 0;
        std::chrono::time_point<std::chrono::system_clock> start = std::chrono::system_clock::now();

        for (int i = 0; i < epoch_; i++) {
            learning_private::forward_propagation(ls);
            learning_private::backward_propagation(ls);
            learning_private::update_params(ls, grad_weight);
            if (i % check_period == 0 && dev_size > 0 && i > 0) {
                learning_private::forward_propagation(ls.dev_inputs, net, ls.dev_hidden_layers);
                mkl_BLAS_impl::mat predictions;
                learning_private::get_predictions(ls.dev_hidden_layers[ls.dev_hidden_layers.size() - 1],
                                                  predictions);
//                if(error < best_error){
//                    stagnation_count = 0;
//                    best_error = error;
//                }else{
//                    stagnation_count++;
//                    if(stagnation_count > patience){
//                        grad_weight *= decay_factor;
//                        stagnation_count = 0;
//                        cout << "decaying learning rate" << endl;
//                    }
//                }
                if(show_progress) {
                    float accuracy = learning_private::get_accuracy(ls.dev_labels, predictions);
                    auto end = std::chrono::system_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
                    cout << "-----------------------" << endl;
                    cout << "EPOCH: " << i << endl;
                    cout << "ACCURACY: " << accuracy << "%" << endl;
                    cout << "LEARNING RATE: " << grad_weight << endl;
                    cout << "EPOCH ELAPSED TIME: " << duration << " microseconds" << endl;
                    cout << "APPROXIMATE TIME LEFT: " << (float)((epoch_ - i) * duration) / 1000000.f << " seconds" << endl;
                    start = std::chrono::system_clock::now();
                }
            }
        }
        if (sample_size > 0) {
            learning_private::show_accuracy_sample(ls, sample_size);
        }
        ls.data.manual_free();
    }
} // learning