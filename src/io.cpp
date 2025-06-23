#include "simple_conv.h"
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>

namespace simple_conv::io {

    //!!! YOU MUST MANUALLY FREE DATASET DATA
    mkl_BLAS_impl::mat
    load_dataset(const boost::filesystem::path &filename, bool transposed, char delimiter,
                 bool has_header) {

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

    cv::Mat read_img_to_input_layer(const boost::filesystem::path &path, bool invert, bool normalize) {
        using namespace cv;

        if (!exists(path)) {
            std::cerr << "image doesn't exists: " << path << std::endl;
            throw std::exception();
        }

        Mat img = imread(path.c_str(), CV_8U);
        if(invert)
            img = 255 - img;
        if(normalize)
            img.convertTo(img, CV_32F, 1.0 / 255.0);


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