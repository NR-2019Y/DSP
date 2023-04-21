#pragma once
#include <utility>
#include <string>
#include <cassert>
#include <vector>
#include <array>

template<typename InputIt1, typename InputIt2>
int count_equal(InputIt1 first1, InputIt1 last1, InputIt2 first2) {
    int cnt = 0;
    for (; first1 != last1; ++first1, ++first2) {
        cnt += (*first1 == *first2);
    }
    return cnt;
}

std::array<std::vector<uint8_t>, 4> get_mnist_data() {
    std::array<std::vector<uint8_t>, 4> all_data{
        std::vector<uint8_t>(784 * 60000),
        std::vector<uint8_t>(784 * 10000),
        std::vector<uint8_t>(60000),
        std::vector<uint8_t>(10000)
    };
    const char *c_proj_root = std::getenv("PROJ_DSP_ROOT");
    assert(c_proj_root);
    std::string proj_root(c_proj_root);
    std::string ftrain_image = proj_root + "/CV/datasets/mnist/train-images-idx3-ubyte";
    std::string ftest_image = proj_root + "/CV/datasets/mnist/t10k-images-idx3-ubyte";
    std::string ftrain_label = proj_root + "/CV/datasets/mnist/train-labels-idx1-ubyte";
    std::string ftest_label = proj_root + "/CV/datasets/mnist/t10k-labels-idx1-ubyte";
    const char* const all_files[4] = { ftrain_image.c_str(), ftest_image.c_str(), ftrain_label.c_str(), ftest_label.c_str() };
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
