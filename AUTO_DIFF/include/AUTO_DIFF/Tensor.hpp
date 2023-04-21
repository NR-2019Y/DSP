#pragma once

#include <vector>
#include <cassert>
#include <initializer_list>
#include <algorithm>
#include <iterator>
#include <unordered_set>
#include <utility>
#include <numeric>
#include <random>
#include "AUTO_DIFF/MathFunc.hpp"

class TrainTestMode {
    static bool _is_train;
public:
    static void setTrain() { _is_train = true; }
    static void setTest() { _is_train = false; }
    static bool isTrain() { return _is_train; }
};

bool TrainTestMode::_is_train = true;

// 动态图
class TensorNode {
public:
    template<typename Cont> static int calcTotal(const Cont& _shape) {
        int tot = 1;
        for (int u : _shape) tot *= u;
        return tot;
    }
protected:
    std::vector<int> shape;
    std::vector<float> data;
    std::vector<float> grad;
    std::vector<TensorNode*> nodes;
    TensorNode(const std::vector<int>& _shape, const std::vector<float>& _data) : shape(_shape), data(_data) {
        assert(calcTotal(_shape) == _data.size());
    }
    TensorNode(int _total, std::initializer_list<int> _shape) : shape(_shape), data(_total) {
        assert(_total == calcTotal(_shape));
    }
    TensorNode(int _total, const std::vector<int>& _shape) : shape(_shape), data(_total) {
        assert(_total == calcTotal(_shape));
    }
    //TensorNode(int _total, std::initializer_list<int> _shape, const float* _data) : shape(_shape), data(_data, _data + _total) {
    //    assert(_total == calcTotal(_shape));
    //}
    //TensorNode(int _total, const std::vector<int>& _shape, const float* _data) : shape(_shape), data(_data, _data + _total) {
    //    assert(_total == calcTotal(_shape));
    //}
    explicit TensorNode(std::initializer_list<int> _shape) : TensorNode(calcTotal(_shape), _shape) {}
    explicit TensorNode(const std::vector<int>& _shape) : TensorNode(calcTotal(_shape), _shape) {}
    //TensorNode(std::initializer_list<int> _shape, const float* _data) : TensorNode(calcTotal(_shape), _shape, _data) {}
    //TensorNode(const std::vector<int>& _shape, const float* _data) : TensorNode(calcTotal(_shape), _shape, _data) {}
    TensorNode(const TensorNode&) = delete;
    TensorNode(TensorNode&&) = delete;
public:
    bool requiresGrad() const { return !grad.empty(); }
    bool isLeafNode() const { return nodes.empty(); }
    float* getData() { return data.data(); }
    const float* getData() const { return data.data(); }
    const std::vector<float>& getVecData() const { return data; }
    float* getGrad() { return grad.data(); }
    const float* getGrad() const { return grad.data(); }
    const std::vector<float>& getVecGrad() const { return grad; }
    const std::vector<int>& getShape() const { return shape; }
    std::vector<TensorNode*> getNodes() { return nodes; }
    const std::vector<TensorNode*> getNodes() const { return nodes; }
    float getSum() const { return std::reduce(data.begin(), data.end()); }
    std::vector<int> argMax() const {
        assert(shape.size() >= 2);
        int nlast_dim = shape.back();
        std::vector<int> result(total() / nlast_dim);
        const float* pfirst = data.data();
        for (int i = 0; i < result.size(); ++i) {
            const float* plast = pfirst + nlast_dim;
            result[i] = std::max_element(pfirst, plast) - pfirst;
            pfirst = plast;
        }
        return result;
    }
    int ndim() const { return shape.size(); }
    int total() const { return data.size(); }
    virtual void backward_step() = 0;
    virtual ~TensorNode() = default;
    void assign_sub(const float alpha, const float* pvec) {
        cblas_saxpy(data.size(), -alpha, pvec, 1, data.data(), 1);
    }
    void zeroGrad() {
        assert(requiresGrad());
        std::fill(grad.begin(), grad.end(), 0.0f);
    }
    void backward() {
        // DFS 拓扑排序
        assert(requiresGrad());
        assert(TrainTestMode::isTrain());
        grad.assign(grad.size(), 1.0f);
        if (isLeafNode()) return;
        std::unordered_set<TensorNode*> visited;
        std::vector<std::pair<int, TensorNode*>> q;
        std::vector<TensorNode*> curr_path;
        std::vector<TensorNode*> ord_nodes;
        q.emplace_back(0, this);
        while (!q.empty()) {
            auto pobj = q.back();
            int layer = pobj.first;
            TensorNode* nd = pobj.second;
            q.pop_back();
            if (visited.count(nd)) continue;
            while (curr_path.size() > layer) {
                ord_nodes.push_back(curr_path.back());
                curr_path.pop_back();
            }
            curr_path.push_back(nd);
            visited.insert(nd);
            for (TensorNode* nnd : nd->nodes) {
                if (!nnd->isLeafNode() && !visited.count(nnd)) {
                    q.emplace_back(layer + 1, nnd);
                }
            }
        }
        while (!curr_path.empty()) {
            ord_nodes.push_back(curr_path.back());
            curr_path.pop_back();
        }
        for (auto iter = ord_nodes.rbegin(); iter != ord_nodes.rend(); ++iter) {
            TensorNode* nd = *iter;
            assert(!nd->isLeafNode());
            nd->backward_step();
        }
    }
};

class TensorNodeValue :public TensorNode {
public:
    TensorNodeValue(std::vector<int> _shape, bool requires_grad) : TensorNode(_shape) {
        if (requires_grad) grad.assign(data.size(), 0.0f);
    }
    TensorNodeValue(std::initializer_list<int> _shape, bool requires_grad) : TensorNode(_shape) {
        if (requires_grad) grad.assign(data.size(), 0.0f);
    }
    template<typename InputIt> void assign(InputIt iter) {
        data.assign(iter, iter + data.size());
    }
    void assign(std::initializer_list<float> ilist) {
        assert(ilist.size() == data.size());
        data.assign(ilist);
    }
    void assign(const std::vector<float>& vec) {
        assert(vec.size() == data.size());
        data.assign(vec.begin(), vec.end());
    }
    void fill(float value) {
        std::fill(data.begin(), data.end(), value);
    }
    void fillNormal(float mean = 0.0f, float stddev = 0.01f) {
        std::mt19937 gen(std::random_device{}());
        std::normal_distribution<float> dis(mean, stddev);
        for (float& f : data) f = dis(gen);
    }
    void fillUniform(float a, float b) {
        std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dis(a, b);
        for (float& f : data) f = dis(gen);
    }
    void backward_step() override {}
    void resize(int n) { // 调整第0个维度的大小
        assert(!requiresGrad());
        assert(n > 0);
        int new_total = data.size() / shape[0] * n;
        shape[0] = n;
        data.resize(new_total);
    }
};

class TensorDataIter {
    const std::vector<float>& data;
    const std::vector<int>& labels;
    std::vector<int> indexs;
    const int batch_size;
    const int nFlattenFeatures;
    bool init = true;
    int current_index = 0;
    const bool shuffle;
    const bool drop_last;
    std::mt19937 gen;
public:
    TensorNodeValue batch_data;
    std::vector<int> batch_labels;
public:
    static std::vector<int> get_first_batch_shape(const std::vector<int>& total_shape, int _batch_size) {
        std::vector<int> bshape(total_shape);
        bshape[0] = std::min<int>(_batch_size, bshape[0]);
        return bshape;
    }
    TensorDataIter(const std::vector<float>& _data,
        const std::vector<int>& _labels,
        const std::vector<int>& _total_shape,
        int _batch_size,
        bool _shuffle = true,
        bool _drop_last = false) :data(_data), labels(_labels), batch_size(_batch_size), nFlattenFeatures(_data.size() / _labels.size()), shuffle(_shuffle), drop_last(_drop_last), gen(std::random_device{}()), batch_data(get_first_batch_shape(_total_shape, _batch_size), false), batch_labels( std::min<int>(_labels.size(), _batch_size) ) {
        assert(TensorNode::calcTotal(_total_shape) == data.size());
        assert(labels.size() == _total_shape[0]);
        if (!shuffle) {
            assert(!drop_last);
            std::copy(data.begin(), data.begin() + batch_labels.size() * nFlattenFeatures, batch_data.getData());
            std::copy(labels.begin(), labels.begin() + batch_labels.size(), batch_labels.begin());
        }
        else {
            indexs.assign(labels.size(), 0);
            std::iota(indexs.begin(), indexs.end(), 0);
            std::shuffle(indexs.begin(), indexs.end(), gen);
            for (int i = 0, im = 0; i < batch_labels.size(); ++i, im += nFlattenFeatures) {
                int rand_index = indexs[i];
                std::copy_n(data.begin() + rand_index * nFlattenFeatures, nFlattenFeatures, batch_data.getData() + im);
                batch_labels[i] = labels[rand_index];
            }
        }
        current_index += batch_labels.size();
    }
    void next() {
        if (init) {
            init = false;
            return;
        }
        if (current_index >= labels.size() ) {
            current_index = 0;
            if (shuffle) std::shuffle(indexs.begin(), indexs.end(), gen);
        }
        else if (drop_last && current_index + batch_size > labels.size()) {
            current_index = 0;
            assert(shuffle);
            std::shuffle(indexs.begin(), indexs.end(), gen);
        }
        int curr_batch_size = std::min<int>(labels.size() - current_index, batch_size);
        if (curr_batch_size != batch_labels.size()) {
            batch_data.resize(curr_batch_size);
            batch_labels.resize(curr_batch_size);
        }
        if (!shuffle) {
            std::copy_n(data.begin() + current_index * nFlattenFeatures, curr_batch_size * nFlattenFeatures, batch_data.getData());
            std::copy_n(labels.begin() + current_index, curr_batch_size, batch_labels.data());
        }
        else {
            for (int i = 0, im = 0; i < curr_batch_size; ++i, im += nFlattenFeatures) {
                int rand_index = indexs[i + current_index];
                std::copy_n(data.begin() + rand_index * nFlattenFeatures, nFlattenFeatures, batch_data.getData() + im);
                batch_labels[i] = labels[rand_index];
            }
        }
        current_index += curr_batch_size;
    }
    int num_samples_one_epoch() const {
        if (!drop_last) return labels.size();
        return labels.size() - labels.size() % batch_size;
    }
    int iter_times_one_epoch() const {
        if (!drop_last) {
            int q = labels.size() / batch_size;
            int r = labels.size() % batch_size;
            if (r == 0) return q;
            return q + 1;
        }
        return labels.size() / batch_size;
    }
};


class TensorNodeAdd : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t0, TensorNode* t1) {
        assert(t0->getShape() == t1->getShape());
        std::vector<float> result(t0->total());
        vsAdd(t0->total(), t0->getData(), t1->getData(), result.data());
        return result;
    }
    TensorNodeAdd(TensorNode* t0, TensorNode* t1) : TensorNode(t0->getShape(), compute(t0, t1)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t0->requiresGrad() && !t1->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        if (t0->requiresGrad()) nodes.push_back(t0);
        if (t1->requiresGrad()) nodes.push_back(t1);
    }
    void backward_step() override {
        for (auto& nd : nodes) {
            vsAdd(grad.size(), grad.data(), nd->getGrad(), nd->getGrad());
        }
    }
};

class TensorNodeSub : public TensorNode {
    int flag = 0;
public:
    static std::vector<float> compute(TensorNode* t0, TensorNode* t1) {
        assert(t0->getShape() == t1->getShape());
        std::vector<float> result(t0->total());
        vsSub(t0->total(), t0->getData(), t1->getData(), result.data());
        return result;
    }
    TensorNodeSub(TensorNode* t0, TensorNode* t1) :TensorNode(t0->getShape(), compute(t0, t1)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t0->requiresGrad() && !t1->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        if (t0->requiresGrad()) {
            flag |= (1 << 0);
            nodes.push_back(t0);
        }
        if (t1->requiresGrad()) {
            flag |= (1 << 1);
            nodes.push_back(t1);
        }
    }
    void backward_step() override {
        assert(flag != 0);
        if (flag == 1) {
            vsAdd(grad.size(), nodes[0]->getGrad(), grad.data(), nodes[0]->getGrad());
        }
        else if (flag == 2) {
            vsSub(grad.size(), nodes[1]->getGrad(), grad.data(), nodes[1]->getGrad());
        }
        else {
            vsAdd(grad.size(), nodes[0]->getGrad(), grad.data(), nodes[0]->getGrad());
            vsSub(grad.size(), nodes[1]->getGrad(), grad.data(), nodes[1]->getGrad());
        }
    }
};

class TensorNodeMul : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t0, TensorNode* t1) {
        assert(t0->getShape() == t1->getShape());
        std::vector<float> result(t0->total());
        vsMul(t0->total(), t0->getData(), t1->getData(), result.data());
        return result;
    }
    TensorNodeMul(TensorNode* t0, TensorNode* t1) :TensorNode(t0->getShape(), compute(t0, t1)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t0->requiresGrad() && !t1->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.reserve(2);
        nodes.push_back(t0);
        nodes.push_back(t1);
    }
    void backward_step() override {
        if (nodes[0]->requiresGrad()) {
            ts::gmul(grad.size(), grad.data(), nodes[1]->getData(), nodes[0]->getGrad());
        }
        if (nodes[1]->requiresGrad()) {
            ts::gmul(grad.size(), grad.data(), nodes[0]->getData(), nodes[1]->getGrad());
        }
    }
};

class TensorNodeDiv : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t0, TensorNode* t1) {
        assert(t0->getShape() == t1->getShape());
        std::vector<float> result(t0->total());
        vsDiv(t0->total(), t0->getData(), t1->getData(), result.data());
        return result;
    }
    TensorNodeDiv(TensorNode* t0, TensorNode* t1) : TensorNode(t0->getShape(), compute(t0, t1)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t0->requiresGrad() && !t1->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.reserve(2);
        nodes.push_back(t0);
        nodes.push_back(t1);
    }
    void backward_step() override {
        if (nodes[0]->requiresGrad()) {
            ts::gdiv(grad.size(), grad.data(), nodes[1]->getData(), nodes[0]->getGrad());
        }
        if (nodes[1]->requiresGrad()) {
            ts::gdiv_nd1(grad.size(), grad.data(), nodes[0]->getData(), nodes[1]->getData(), nodes[1]->getGrad());
        }
    }
};

class TensorNodeAddScalar : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t, float value) {
        std::vector<float> result(t->getVecData());
        ts::add_scalar(result.size(), value, result.data());
        return result;
    }
    TensorNodeAddScalar(TensorNode* t, float value) : TensorNode(t->getShape(), compute(t, value)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        vsAdd(grad.size(), nodes[0]->getGrad(), grad.data(), nodes[0]->getGrad());
    }
};

class TensorNodeSubLScalar : public TensorNode {
public:
    static std::vector<float> compute(float value, TensorNode* t) {
        std::vector<float> result(t->getVecData());
        ts::sub_lscalar(result.size(), value, result.data());
        return result;
    }
    TensorNodeSubLScalar(float value, TensorNode* t) : TensorNode(t->getShape(), compute(value, t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        vsSub(grad.size(), nodes[0]->getGrad(), grad.data(), nodes[0]->getGrad());
    }
};

class TensorNodeMulScalar : public TensorNode {
    const float scalar_value;
public:
    static std::vector<float> compute(TensorNode* t, float value) {
        std::vector<float> result(t->getVecData());
        cblas_sscal(result.size(), value, result.data(), 1);
        return result;
    }
    TensorNodeMulScalar(TensorNode* t, float value) : TensorNode(t->getShape(), compute(t, value)), scalar_value(value) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        cblas_saxpy(grad.size(), scalar_value, grad.data(), 1, nodes[0]->getGrad(), 1);
    }
};

class TensorNodeDivLScalar : public TensorNode {
    const float scalar_value;
public:
    static std::vector<float> compute(float value, TensorNode* t) {
        std::vector<float> result(t->total());
        vsInv(result.size(), t->getData(), result.data());
        cblas_sscal(result.size(), value, result.data(), 1);
        return result;
    }
    TensorNodeDivLScalar(float value, TensorNode* t) :TensorNode(t->getShape(), compute(value, t)), scalar_value(value) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        ts::gdiv_nd1_scalar(grad.size(), grad.data(), scalar_value, nodes[0]->getData(), nodes[0]->getGrad());
    }
};

class TensorNodeNeg : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t) {
        std::vector<float> result(t->getVecData());
        ts::ineg(result.size(), result.data());
        return result;
    }
    TensorNodeNeg(TensorNode* t) : TensorNode(t->getShape(), compute(t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        vsSub(grad.size(), nodes[0]->getGrad(), grad.data(), nodes[0]->getGrad());
    }
};

class TensorNodeExp :public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t) {
        std::vector<float> result(t->total());
        vsExp(result.size(), t->getData(), result.data());
        return result;
    }
    TensorNodeExp(TensorNode* t) : TensorNode(t->getShape(), compute(t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        ts::gmul(grad.size(), grad.data(), data.data(), nodes[0]->getGrad());
    }
};

class TensorNodeExpNeg : public TensorNode {
    // Y = exp(-X)
public:
    static std::vector<float> compute(TensorNode* t) {
        std::vector<float> result(t->total());
        ts::expneg(result.size(), t->getData(), result.data());
        return result;
    }
    TensorNodeExpNeg(TensorNode* t) : TensorNode(t->getShape(), compute(t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        ts::gmul_isub(grad.size(), grad.data(), data.data(), nodes[0]->getGrad());
    }
};

class TensorNodeSquare : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t) {
        std::vector<float> result(t->total());
        vsSqr(result.size(), t->getData(), result.data());
        return result;
    }
    TensorNodeSquare(TensorNode* t) :TensorNode(t->getShape(), compute(t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        ts::gmul_alpha(grad.size(), 2.0f, grad.data(), nodes[0]->getData(), nodes[0]->getGrad());
    }
};

class TensorNodeMatMul : public TensorNode {
public:
    static std::vector<int> computeOutShape(const std::vector<int>& shape0, const std::vector<int>& shape1) {
        assert(shape0.size() >= 2 && shape1.size() == 2 && shape0.back() == shape1[0]);
        std::vector<int> outShape(shape0);
        outShape.back() = shape1.back();
        return outShape;
    }
    static std::vector<float> compute(TensorNode* t0, TensorNode* t1) {
        const int M = t0->total() / t0->getShape().back();
        const int K = t0->getShape().back();
        const int N = t1->getShape().back();
        std::vector<float> result(M * N);
        const int LDA = K;
        const int LDB = N;
        const int LDC = N;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, t0->getData(), LDA, t1->getData(), LDB, 0.0f, result.data(), LDC);
        return result;
    }
    TensorNodeMatMul(TensorNode* t0, TensorNode* t1) : TensorNode(computeOutShape(t0->getShape(), t1->getShape()), compute(t0, t1)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t0->requiresGrad() && !t1->requiresGrad()) return;
        grad.assign(data.size(), 0);
        nodes.assign({ t0, t1 });
    }
    void backward_step() override {
        const int M = nodes[0]->total() / nodes[0]->getShape().back();
        const int K = nodes[0]->getShape().back();
        const int N = nodes[1]->getShape().back();
        if (nodes[0]->requiresGrad()) {
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0f, grad.data(), N, nodes[1]->getData(), N, 1.0f, nodes[0]->getGrad(), K);
        }
        if (nodes[1]->requiresGrad()) {
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0f, nodes[0]->getData(), K, grad.data(), N, 1.0f, nodes[1]->getGrad(), N);
        }
    }
};

class TensorNodeAddBias : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t0, TensorNode* t1) {
        assert(t0->ndim() >= 2 && t1->ndim() == 1);
        assert(t0->getShape().back() == t1->total());
        std::vector<float> result(t0->total());
        const int M = t0->total() / t1->total();
        const int N = t1->total();
        for (int i = 0; i < M; ++i) {
            vsAdd(N, t0->getData() + i * N, t1->getData(), result.data() + i * N);
        }
    }
    TensorNodeAddBias(TensorNode* t0, TensorNode* t1) :TensorNode(t0->getShape(), compute(t0, t1)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t0->requiresGrad() && !t1->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.assign({ t0, t1 });
    }
    void backward_step() override {
        if (nodes[0]->requiresGrad()) {
            vsAdd(grad.size(), nodes[0]->getGrad(), grad.data(), nodes[0]->getGrad());
        }
        if (nodes[1]->requiresGrad()) {
            const int M = nodes[0]->total() / nodes[1]->total();
            const int N = nodes[1]->total();
            float* pgnd1 = nodes[1]->getGrad();
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < M; ++i) {
                    pgnd1[j] += grad[i * N + j];
                }
            }
        }
    }
};

class TensorNodeLinear : public TensorNode {
public:
    static std::vector<int> computeOutShape(TensorNode* X, TensorNode* W, TensorNode* b) {
        assert(X->ndim() >= 2 && W->ndim() == 2 && b->ndim() == 1);
        assert(X->getShape().back() == W->getShape()[0] && W->getShape()[1] == b->total());
        std::vector<int> outShape(X->getShape());
        outShape.back() = b->total();
        return outShape;
    }
    static std::vector<float> compute(TensorNode* X, TensorNode* W, TensorNode* b) {
        const int M = X->total() / X->getShape().back();
        const int K = X->getShape().back();
        const int N = b->total();
        std::vector<float> result(M * N);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0f, X->getData(), K, W->getData(), N, 0.0f, result.data(), N);
        for (int i = 0; i < M; ++i) {
            vsAdd(N, result.data() + i * N, b->getData(), result.data() + i * N);
        }
        return result;
    }
    TensorNodeLinear(TensorNode* X, TensorNode* W, TensorNode* b) :TensorNode(computeOutShape(X, W, b), compute(X, W, b)) {
        if (!TrainTestMode::isTrain()) return;
        if (!X->requiresGrad() && !W->requiresGrad() && !b->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.assign({ X, W, b });
    }
    void backward_step() override {
        const int M = nodes[0]->total() / nodes[0]->getShape().back();
        const int K = nodes[0]->getShape().back();
        const int N = nodes[2]->total();
        if (nodes[0]->requiresGrad()) { // (M, N) (K, N)T => (M, K)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, K, N, 1.0f, grad.data(), N, nodes[1]->getData(), N, 1.0f, nodes[0]->getGrad(), K);
        }
        if (nodes[1]->requiresGrad()) { // (M, K)T, (M, N) => (K, N)
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, K, N, M, 1.0f, nodes[0]->getData(), K, grad.data(), N, 1.0f, nodes[1]->getGrad(), N);
        }
        if (nodes[2]->requiresGrad()) {
            float* pgb = nodes[2]->getGrad();
            for (int j = 0; j < N; ++j) {
                for (int i = 0; i < M; ++i) {
                    pgb[j] += grad[i * N + j];
                }
            }
        }
    }
};

class TensorNodeSigmoid :public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t) {
        std::vector<float> result(t->total());
        const float* x = t->getData();
        for (int i = 0; i < result.size(); ++i) {
            result[i] = 1.0f / (1.0f + std::expf(-x[i]));
        }
        return result;
    }
    TensorNodeSigmoid(TensorNode* t) : TensorNode(t->getShape(), compute(t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        float* pgnd = nodes[0]->getGrad();
        for (int i = 0; i < grad.size(); ++i) {
            pgnd[i] += grad[i] * data[i] * (1.0f - data[i]);
        }
    }
};

// reduce最后一个维度
class TensorNodeReduceSumLastDim :public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t) {
        const int nlast_dim = t->getShape().back();
        std::vector<float> result(t->total() / nlast_dim);
        const float* pfirst = t->getData();
        for (int i = 0; i < result.size(); ++i) {
            const float* plast = pfirst + nlast_dim;
            result[i] = std::reduce(pfirst, plast);
            pfirst = plast;
        }
        return result;
    }
    TensorNodeReduceSumLastDim(TensorNode* t) : TensorNode(std::vector<int>(t->getShape().begin(), t->getShape().end() - 1), compute(t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        const int nlast_dim = nodes[0]->getShape().back();
        const int count = grad.size();
        for (int i = 0; i < nlast_dim; ++i) {
            cblas_saxpy(count, 1.0, grad.data(), 1, nodes[0]->getGrad() + i, nlast_dim);
        }
    }
};

class TensorNodeBinaryCrossEntroyLoss : public TensorNode {
    const std::vector<int>& labels;
    std::vector<float> pred_proba;
public:
    // from logits
    void compute(TensorNode* t) {
        assert(t->getShape().back() == 1);
        pred_proba.assign(t->total(), 0.0f);
        const float* px = t->getData();
        for (int i = 0; i < pred_proba.size(); ++i) {
            pred_proba[i] = 1.0f / (1.0f + std::expf(-px[i]));
        }
        float scale = 1.0f / static_cast<float>(t->total());
        for (int i = 0; i < data.size(); ++i) {
            if (labels[i]) {
                data[i] = -scale * std::logf(pred_proba[i]);
            }
            else {
                data[i] = -scale * std::logf(1.0 - pred_proba[i]);
            }
        }
    }
    const std::vector<float>& getProba() const {
        return pred_proba;
    }
    TensorNodeBinaryCrossEntroyLoss(TensorNode* t, const std::vector<int>& _labels) : TensorNode(std::vector<int>(t->getShape().begin(), t->getShape().end() - 1)), labels(_labels) {
        compute(t);
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        float* pgnd = nodes[0]->getGrad();
        float scale = 1.0f / static_cast<float>(nodes[0]->total());
        for (int i = 0; i < grad.size(); ++i) {
            pgnd[i] += scale * grad[i] * (pred_proba[i] - labels[i]);
        }
    }
};

class TensorNodeSparseCrossEntroyLoss : public TensorNode {
    const std::vector<int>& labels;
    std::vector<float> pred_proba;
public:
    // from logits
    void compute(TensorNode* t) {
        const int last_dim = t->getShape().back();
        std::vector<float> logits_sc(t->getVecData());
        const int trows = t->total() / last_dim;
        for (float* pfirst = logits_sc.data(), *plast = logits_sc.data() + logits_sc.size(); pfirst != plast; pfirst += last_dim) {
            float curr_max = *std::max_element(pfirst, pfirst + last_dim);
            for (int i = 0; i < last_dim; ++i) {
                pfirst[i] -= curr_max;
            }
        }
        std::vector<float>& logits_sc_exp = pred_proba;
        logits_sc_exp.assign(logits_sc.size(), 0.0f);
        vsExp(logits_sc_exp.size(), logits_sc.data(), logits_sc_exp.data());
        float scale = 1.0f / static_cast<float>(trows);
        for (int i = 0, ilab = 0; i < logits_sc.size(); i += last_dim, ++ilab) {
            float se = std::reduce(logits_sc_exp.data() + i, logits_sc_exp.data() + i + last_dim);
            float lse = std::logf(se);
            data[ilab] = scale * (lse - logits_sc[i + labels[ilab]]);
            float inv_se = 1.0f / se;
            for (int j = 0; j < last_dim; ++j) {
                logits_sc_exp[i + j] *= inv_se;
            }
        }
    }
    const std::vector<float>& getProba() const {
        return pred_proba;
    }
    TensorNodeSparseCrossEntroyLoss(TensorNode* t, const std::vector<int>& _labels) : TensorNode(std::vector<int>(t->getShape().begin(), t->getShape().end() - 1)), labels(_labels) {
        compute(t);
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0);
        nodes.push_back(t);
    }
    void backward_step() override {
        const int last_dim = nodes[0]->getShape().back();
        const int trows = grad.size();
        float scale = 1.0f / static_cast<float>(trows);
        float* pgnd = nodes[0]->getGrad();
        for (int i = 0, ilab = 0; ilab < trows; i += last_dim, ++ilab) {
            int curr_label = labels[ilab];
            float sc = scale * grad[ilab];
            for (int j = 0; j < last_dim; ++j) {
                if (j == curr_label) {
                    pgnd[i + j] += sc * (pred_proba[i + j] - 1.0f);
                }
                else {
                    pgnd[i + j] += sc * pred_proba[i + j];
                }
            }
        }
    }
};

class TensorNodeRelu : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t) {
        std::vector<float> result(t->total());
        const float* x = t->getData();
        for (int i = 0; i < result.size(); ++i) result[i] = std::max<float>(x[i], 0.0f);
        return result;
    }
    TensorNodeRelu(TensorNode* t) : TensorNode(t->getShape(), compute(t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        float* pgnd = nodes[0]->getGrad();
        for (int i = 0; i < grad.size(); ++i) pgnd[i] += grad[i] * (data[i] > 0.0f);
    }
};

class TensorNodeLog : public TensorNode {
public:
    static std::vector<float> compute(TensorNode* t) {
        std::vector<float> result(t->total());
        vsLn(result.size(), t->getData(), result.data());
        return result;
    }
    TensorNodeLog(TensorNode* t) : TensorNode(t->getShape(), compute(t)) {
        if (!TrainTestMode::isTrain()) return;
        if (!t->requiresGrad()) return;
        grad.assign(data.size(), 0.0f);
        nodes.push_back(t);
    }
    void backward_step() override {
        ts::gdiv(grad.size(), grad.data(), nodes[0]->getData(), nodes[0]->getGrad());
    }
};

struct Tensor {
    TensorNode* pnode;
    std::unordered_set<TensorNode*>* pndset;
    explicit Tensor(TensorNode* ptr, std::unordered_set<TensorNode*>* _pndset, bool add_to_set = true) : pnode(ptr), pndset(_pndset) {
        if (add_to_set) {
            pndset->insert(ptr);
        }
    }
    friend Tensor operator+(const Tensor& tensor0, const Tensor& tensor1) {
        assert(tensor0.pndset == tensor1.pndset);
        return Tensor(new TensorNodeAdd(tensor0.pnode, tensor1.pnode), tensor0.pndset);
    }
    friend Tensor operator+(const Tensor& tensor, float value) {
        return Tensor(new TensorNodeAddScalar(tensor.pnode, value), tensor.pndset);
    }
    friend Tensor operator+(float value, const Tensor& tensor) {
        return Tensor(new TensorNodeAddScalar(tensor.pnode, value), tensor.pndset);
    }
    friend Tensor operator-(const Tensor& tensor0, const Tensor& tensor1) {
        assert(tensor0.pndset == tensor1.pndset);
        return Tensor(new TensorNodeSub(tensor0.pnode, tensor1.pnode), tensor0.pndset);
    }
    friend Tensor operator-(const Tensor& tensor, float value) {
        return Tensor(new TensorNodeAddScalar(tensor.pnode, -value), tensor.pndset);
    }
    friend Tensor operator-(float value, const Tensor& tensor) {
        return Tensor(new TensorNodeSubLScalar(value, tensor.pnode), tensor.pndset);
    }
    friend Tensor operator*(const Tensor& tensor0, const Tensor& tensor1) {
        assert(tensor0.pndset == tensor1.pndset);
        return Tensor(new TensorNodeMul(tensor0.pnode, tensor1.pnode), tensor0.pndset);
    }
    friend Tensor operator*(const Tensor& tensor, float value) {
        return Tensor(new TensorNodeMulScalar(tensor.pnode, value), tensor.pndset);
    }
    friend Tensor operator*(float value, const Tensor& tensor) {
        return Tensor(new TensorNodeMulScalar(tensor.pnode, value), tensor.pndset);
    }
    friend Tensor operator/(const Tensor& tensor0, const Tensor& tensor1) {
        assert(tensor0.pndset == tensor1.pndset);
        return Tensor(new TensorNodeDiv(tensor0.pnode, tensor1.pnode), tensor0.pndset);
    }
    friend Tensor operator/(const Tensor& tensor, float value) {
        return Tensor(new TensorNodeMulScalar(tensor.pnode, 1.0f / value), tensor.pndset);
    }
    friend Tensor operator/(float value, const Tensor& tensor) {
        return Tensor(new TensorNodeDivLScalar(value, tensor.pnode), tensor.pndset);
    }
    Tensor operator-() {
        return Tensor(new TensorNodeNeg(pnode), pndset);
    }
    Tensor exp() const {
        return Tensor(new TensorNodeExp(pnode), pndset);
    }
    Tensor expneg() const {
        return Tensor(new TensorNodeExpNeg(pnode), pndset);
    }
    Tensor square() const {
        return Tensor(new TensorNodeSquare(pnode), pndset);
    }
    void backward() {
        pnode->backward();
    }
    const std::vector<float>& getVecData() const { return pnode->getVecData(); }
    const std::vector<float>& getVecGrad() const { return pnode->getVecGrad(); }
    Tensor matmul(const Tensor& tensor) const {
        assert(pndset == tensor.pndset);
        return Tensor(new TensorNodeMatMul(pnode, tensor.pnode), pndset);
    }
    Tensor addBias(const Tensor& tensor) const {
        assert(pndset == tensor.pndset);
        return Tensor(new TensorNodeAddBias(pnode, tensor.pnode), pndset);
    }
    static Tensor linear(const Tensor& X, const Tensor& W, const Tensor& b) {
        assert(X.pndset == W.pndset && X.pndset == b.pndset);
        return Tensor(new TensorNodeLinear(X.pnode, W.pnode, b.pnode), X.pndset);
    }
    Tensor sigmoid() const {
        return Tensor(new TensorNodeSigmoid(pnode), pndset);
    }
    Tensor log() const {
        return Tensor(new TensorNodeLog(pnode), pndset);
    }
    Tensor relu() const {
        return Tensor(new TensorNodeRelu(pnode), pndset);
    }
};

void test_sigmoid() {
    std::unordered_set<TensorNode*> ndset;
    TensorNodeValue xtensor({ 4 }, true);
    xtensor.assign({ -2.3f, -1.2f, 1.7f, 4.0f });
    Tensor X(&xtensor, &ndset, false);
    std::cout << "ori data x:";
    for (float f : X.getVecData()) std::cout << ' ' << f;
    std::cout << '\n';
    //Tensor Y = 1.0f / (1.0f + X.expneg());
    Tensor Y = 1.0f / (1.0f + (-X).exp());
    Y.backward();
    for (float f : Y.getVecData()) std::cout << ' ' << f;
    std::cout << '\n';
    std::cout << "grad T: ";
    for (float f : Y.getVecData()) std::cout << ' ' << f * (1 - f);
    std::cout << '\n';
    std::cout << "grad x: ";
    for (float f : X.getVecGrad()) std::cout << ' ' << f;
    std::cout << '\n';
    std::cout << "data x: ";
    for (float f : X.getVecData()) std::cout << ' ' << f;
    std::cout << '\n';
    for (TensorNode* p : ndset) delete p;
}

void test_f2() {
    std::unordered_set<TensorNode*> ndset;
    TensorNodeValue xtensor({ 5 }, true);
    xtensor.assign({ -2.0f, -1.0f, 0.0f, 1.0f, 2.0f });
    Tensor X(&xtensor, &ndset, false);
    Tensor Y = 3.1f + 3.5f / (X.square() + 1.0f) + X.square() * 4.7f - X * 3.2f + 7.6 * X.square() + 5.0f;
    Y.backward();
    std::cout << "Y data: ";
    for (float f : Y.getVecData()) std::cout << ' ' << f;
    std::cout << '\n';
    std::cout << "Y grad: ";
    for (float f : X.getVecGrad()) std::cout << ' ' << f;
    std::cout << '\n';
    for (TensorNode* p : ndset) delete p;
}
