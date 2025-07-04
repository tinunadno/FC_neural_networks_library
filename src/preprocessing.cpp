#include "simple_conv.h"

namespace simple_conv::preprocessing{
    void crop_image(mkl_BLAS_impl::mat& img){
        cv::Mat wrapper(img.rows, img.cols, CV_32F, img.data);
        cv::Mat copy = wrapper.clone();
        threshold(copy,copy, 120, 255, cv::THRESH_BINARY);
        Canny(copy, copy, 120, 200);

        std::vector<std::vector<cv::Point>> contours;

        findContours(copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
        std::vector<cv::Point2f> flat_contour;
        for(auto& i : contours){
            flat_contour.insert(flat_contour.begin(), i.begin(), i.end());
        }

        auto roi = cv::boundingRect(flat_contour);

        int diff = abs(roi.width - roi.height) / 2;
        if(roi.width < roi.height){
            roi.x -= diff;
            roi.width = roi.height;
        }else{
            roi.y -= diff;
            roi.height = roi.width;
        }

        wrapper = wrapper(roi);
        cv::resize(wrapper, wrapper,cv::Size(28, 28), cv::INTER_LINEAR);

        img.manual_free();
        img = mkl_BLAS_impl::mat(wrapper);
        img.reshape(1, (int) img.size());
    }


    void crop_image(cv::Mat& img, bool blur_me, int blur_size){
        if(img.type() == CV_8UC3){
            cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
        }
        if(img.type() == CV_32F){
            img.convertTo(img, CV_8UC1, 255.0);
        }
        if(blur_me){
            cv::blur(img, img, cv::Size(blur_size, blur_size));
        }

        cv::Mat copy = img.clone();
        threshold(copy,copy, 120, 255, cv::THRESH_BINARY);
        Canny(copy, copy, 120, 200);

        std::vector<std::vector<cv::Point>> contours;

        findContours(copy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
        std::vector<cv::Point2f> flat_contour;
        for(auto& i : contours){
            flat_contour.insert(flat_contour.begin(), i.begin(), i.end());
        }

        auto roi = cv::boundingRect(flat_contour);


        int diff = abs(roi.width - roi.height) / 2;
        if(roi.width < roi.height){
            roi.x -= diff;
            roi.x = std::max(roi.x, 0);
            roi.width = roi.height;
            roi.width = std::min(img.size().width - roi.x, roi.width);
        }else{
            roi.y -= diff;
            roi.y = std::max(roi.y, 0);
            roi.height = roi.width;
            roi.height = std::min(img.size().height - roi.y, roi.height);
        }

        if(roi.width == 0 || roi.height == 0){
            return;
        }

        img = img(roi);

        img.convertTo(img, CV_32F, 1.0 / 255.0);
        cv::resize(img, img,cv::Size(28, 28), cv::INTER_LINEAR);
    }

    void preprocess_dataset(const boost::filesystem::path& in, const boost::filesystem::path& out){
        if(!exists(in)){
            throw std::runtime_error("dataset doesn't exists!");
        }
        auto dataset = simple_conv::io::load_dataset(in, false, ',', true);
        int size = (int)sqrtf((float)dataset.cols);
        for(int i = 0; i < dataset.rows; i++){
            cv::Mat row(1, dataset.cols - 1, CV_32F, dataset.data + dataset.cols * i + 1);
            row = row.reshape(1, {size, size});
            crop_image(row, true, 2);
            row = row.reshape(1, {1, (int) row.total()});
            mempcpy(dataset.data + dataset.cols * i + 1, row.data, (dataset.cols - 1) * sizeof(float));
            cv::Mat temp(1, dataset.cols - 1, CV_32F, dataset.data + dataset.cols * i + 1);
            int a =0;
            (void)a;
        }

        simple_conv::io::save_dataset(out, dataset, ',');

    }
}