#include <cstdio>
#include <cstdint>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <string>
#include <algorithm>
#include <utility>


std::array<std::vector<uint8_t>, 4> get_train_test_data() {
    std::array<std::vector<uint8_t>, 4> all_data{
        std::vector<uint8_t>(784 * 60000),
        std::vector<uint8_t>(784 * 10000),
        std::vector<uint8_t>(60000),
        std::vector<uint8_t>(10000)
    };
    const char* ftrain_image = "../datasets/mnist/train-images-idx3-ubyte";
    const char* ftest_image = "../datasets/mnist/t10k-images-idx3-ubyte";
    const char* ftrain_label = "../datasets/mnist/train-labels-idx1-ubyte";
    const char* ftest_label = "../datasets/mnist/t10k-labels-idx1-ubyte";
    const char* const all_files[4] = { ftrain_image, ftest_image, ftrain_label, ftest_label };
    const int offset_list[4] = { 16, 16, 8, 8 };
    for (int i = 0; i < 4; ++i) {
        FILE* f;
        fopen_s(&f, all_files[i], "rb");
        fseek(f, offset_list[i], SEEK_SET);
        size_t sz = fread(all_data[i].data(), sizeof(uint8_t), all_data[i].size(), f);
        //std::cout <<  all_data[i].size() << ", sz = " << sz << '\n';
        fclose(f);
    }
    return all_data;
}

typedef std::pair<cv::Mat_<float>, cv::Mat_<float>> pair_mat_t;
pair_mat_t get_params() {
    cv::Mat_<float> W(784, 10);
    cv::randn(W, 0.0f, 0.01f);
    cv::Mat_<float> b = cv::Mat_<float>::zeros(1, 10);
    return std::make_pair(W, b);
}

cv::Mat_<float> model(cv::Mat_<float> X, pair_mat_t params) {
    auto [W, b] = params;
    return X * W + cv::repeat(b, X.rows, 1);
}

cv::Mat_<float> softmax(cv::Mat_<float> logits) {
    cv::Mat_<float> nmax;
    cv::reduce(logits, nmax, 1, cv::REDUCE_MAX);
    cv::Mat_<float> logits_sc = logits - cv::repeat(nmax, 1, logits.cols);
    cv::exp(logits_sc, logits_sc);
    cv::Mat_<float> nsum;
    cv::reduce(logits_sc, nsum, 1, cv::REDUCE_SUM);
    return logits_sc / cv::repeat(nsum, 1, logits.cols);
}

float loss_fn(cv::Mat_<float> proba, cv::Mat_<uint8_t> labels) {
    const int n_samples = proba.rows;
    cv::Mat_<float> selected_proba(n_samples, 1);
    for (int i = 0; i < n_samples; ++i) {
        selected_proba(i) = proba(i, +labels(i));
    }
    cv::log(selected_proba, selected_proba);
    return -cv::mean(selected_proba)(0);
}

pair_mat_t grad_params(cv::Mat_<float> X, cv::Mat_<float> proba, cv::Mat_<uint8_t> labels) {
    const int n_samples = proba.rows;
    cv::Mat_<float> p = proba.clone();
    for (int i = 0; i < n_samples; ++i) {
        p(i, +labels(i)) -= 1.0f;
    }
    float n_scale = 1.0f / n_samples;
    cv::Mat_<float> dW = n_scale * (X.t() * p);
    cv::Mat_<float> db;
    cv::reduce(p, db, 0, cv::REDUCE_AVG);
    return std::make_pair(dW, db);
}

double accuracy(cv::Mat_<float> logits, cv::Mat_<uint8_t> labels) {
    const int n_samples = logits.rows;
    cv::Mat_<int32_t> pred;
    cv::reduceArgMax(logits, pred, 1);
    int neq = 0;
    for (int i = 0; i < n_samples; ++i) neq += (pred(i) == labels(i));
    return static_cast<double>(neq) / logits.rows;
}

void SGD(pair_mat_t params, pair_mat_t grads, double lr) {
    auto [W, b] = params;
    auto [dW, db] = grads;
    W -= lr * dW;
    b -= lr * db;
}

int main(int argc, char** agrv) {
    std::array<std::vector<uint8_t>, 4>  train_test_data = get_train_test_data();
    cv::Mat_<uint8_t> ori_train_img(60000, 784, train_test_data[0].data());
    cv::Mat_<uint8_t> ori_test_img(10000, 784, train_test_data[1].data());
    cv::Mat_<uint8_t> train_label(60000, 1, train_test_data[2].data());
    cv::Mat_<uint8_t> test_label(10000, 1, train_test_data[3].data());
    cv::Mat_<float> train_img, test_img;
    ori_train_img.convertTo(train_img, -1, 1. / 255.);
    ori_test_img.convertTo(test_img, -1, 1. / 255.);

    pair_mat_t params = get_params();
    int epochs = 500;

    const int num_train = 60000;
    const int num_test = 10000;
    const int batch_size = 64;
    double lr = 0.01;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        float train_loss = 0.0f;
        double train_acc = 0.0;
        for (int i = 0; i < num_train; i += batch_size) {
            int end_index = std::min(i + batch_size, num_train);
            cv::Mat_<float> xi = train_img.rowRange(i, end_index);
            cv::Mat_<float> yi = train_label.rowRange(i, end_index);
            cv::Mat_<float> train_logits = model(xi, params);
            cv::Mat_<float> train_proba = softmax(train_logits);
            float loss = loss_fn(train_proba, yi);
            auto grads = grad_params(xi, train_proba, yi);
            SGD(params, grads, lr);

            train_loss += loss * xi.rows;
            train_acc += accuracy(train_logits, yi) * xi.rows;
        }
        train_loss /= num_train;
        train_acc /= num_train;

        cv::Mat_<float> test_logits = model(test_img, params);
        double test_acc = accuracy(test_logits, test_label);
        std::cout << "[" << epoch << "]\tTRAIN_LOSS:" << train_loss << "\tTRAIN_ACC:" << train_acc << "\tTEST_ACC:" << test_acc << std::endl;
    }
}

