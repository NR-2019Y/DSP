#pragma once
#include <mkl.h>
#include <cmath>

namespace ts {
    constexpr double eps = 1e-10;
    // Y = exp(-X)
    void expneg(const int n, const float* x, float* y) {
        for (int i = 0; i < n; ++i) {
            y[i] = std::expf(-x[i]);
        }
    }

    void ineg(const int n, float* x) {
        for (int i = 0; i < n; ++i) {
            x[i] = -x[i];
        }
    }

    // Z += X * Y
    void gmul(const int n, const float* x, const float* y, float* z) {
        for (int i = 0; i < n; ++i) {
            z[i] += x[i] * y[i];
        }
    }

    // Z -= X * Y
    void gmul_isub(const int n, const float* x, const float* y, float* z) {
        for (int i = 0; i < n; ++i) {
            z[i] -= x[i] * y[i];
        }
    }

    // Z += alpha * X * Y
    void gmul_alpha(const int n, const float alpha, const float* x, const float* y, float* z) {
        for (int i = 0; i < n; ++i) {
            z[i] += alpha * x[i] * y[i];
        }
    }

    // Z += X / Y
    void gdiv(const int n, const float* x, const float* y, float* z) {
        for (int i = 0; i < n; ++i) {
            z[i] += x[i] / y[i];
        }
    }

    // d1 += grad * v0 / (v1 * v1)
    void gdiv_nd1(const int n, const float* grad, const float* v0, const float* v1, float* d1) {
        for (int i = 0; i < n; ++i) {
            d1[i] -= grad[i] * v0[i] / (v1[i] * v1[i] + eps);
        }
    }

    void gdiv_nd1_scalar(const int n, const float* grad, const float val0, const float* v1, float* d1) {
        for (int i = 0; i < n; ++i) {
            d1[i] -= val0 * grad[i] / (v1[i] * v1[i] + eps);
        }
    }

    void add_scalar(const int n, const float value, float* x) {
        for (int i = 0; i < n; ++i) x[i] += value;
    }

    void sub_lscalar(const int n, const float value, float* x) {
        for (int i = 0; i < n; ++i) x[i] = value - x[i];
    }
}
