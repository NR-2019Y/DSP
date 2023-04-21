#include <iostream>
#include <array>
#include "AUTO_DIFF/Tensor.hpp"
#include "AUTO_DIFF/utils.hpp"

class Model {
    static constexpr int n_hiddens = 256;
    TensorNodeValue tnW1, tnb1, tnW2, tnb2;
    std::unordered_set<TensorNode*> ndset;
public:
    std::array<TensorNode* const, 4> params;
    Model() : tnW1({ 784, n_hiddens }, true),
        tnb1({ n_hiddens }, true),
        tnW2({ n_hiddens, 10 }, true),
        tnb2({ 10 }, true),
        params{ &tnW1, &tnb1, &tnW2, &tnb2 } {
        tnW1.fillNormal();
        tnW2.fillNormal();
    }
    Tensor call(TensorNode* tnX) {
        Tensor X(tnX, &ndset, false);
        Tensor W1(&tnW1, &ndset, false);
        Tensor b1(&tnb1, &ndset, false);
        Tensor W2(&tnW2, &ndset, false);
        Tensor b2(&tnb2, &ndset, false);
        Tensor A1 = Tensor::linear(X, W1, b1).relu();
        Tensor logits = Tensor::linear(A1, W2, b2);
        return logits;
    }
    void clear() {
        for (TensorNode* p : ndset) delete p;
        ndset.clear();
    }
    ~Model() {
        for (TensorNode* p : ndset) delete p;
    }
};

int main() {
    auto all_data = get_mnist_data();
    std::vector<float> train_x(all_data[0].size());
    for (int i = 0; i < train_x.size(); ++i) train_x[i] = (float)all_data[0][i] / 255.0f;
    std::vector<int> train_y(all_data[2].begin(), all_data[2].end());

    Model model;
    TensorDataIter train_iter(train_x, train_y, { 60000, 784 }, 64, true, false);
    TensorNodeValue test_data({ 10000, 784 }, false);
    for (int i = 0; i < test_data.total(); ++i) {
        test_data.getData()[i] = (float)all_data[1][i] / 255.0f;
    }
    std::vector<int> test_y(all_data[3].begin(), all_data[3].end());

    constexpr int num_epochs = 1000;
    const int niter_train_per_epoch = train_iter.iter_times_one_epoch();
    const int train_samples_per_epoch = train_iter.num_samples_one_epoch();
    const int test_samples_per_epoch = test_y.size();
    const float learning_rate = 0.01;
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        double total_train_loss = 0.0;
        double total_train_nacc = 0.0;
        TrainTestMode::setTrain();
        for (int i = 0; i < niter_train_per_epoch; ++i) {
            for (TensorNode* p : model.params) p->zeroGrad();
            train_iter.next();
            Tensor logits = model.call(&train_iter.batch_data);
            TensorNodeSparseCrossEntroyLoss ndloss(logits.pnode, train_iter.batch_labels);
            ndloss.backward();

            const int current_batch_size = train_iter.batch_labels.size();
            total_train_loss += ndloss.getSum() * current_batch_size;
            std::vector<int> batch_pred = logits.pnode->argMax();
            total_train_nacc += count_equal(batch_pred.begin(), batch_pred.end(), train_iter.batch_labels.begin());

            for (TensorNode* p : model.params) p->assign_sub(learning_rate, p->getGrad());
            model.clear();
        }
        double train_loss = total_train_loss / train_samples_per_epoch;
        double train_acc = total_train_nacc / train_samples_per_epoch;

        TrainTestMode::setTest();
        Tensor test_logits = model.call(&test_data);
        TensorNodeSparseCrossEntroyLoss test_ndloss(test_logits.pnode, test_y);
        double test_loss = test_ndloss.getSum();
        std::vector<int> test_pred = test_logits.pnode->argMax();
        double test_acc = static_cast<double>(count_equal(test_pred.begin(), test_pred.end(), test_y.begin())) / test_samples_per_epoch;
        std::cout << "epoch: " << epoch << "\tLOSS: " << train_loss << "\tACC: " << train_acc << "\tTEST_LOSS: " << test_loss << "\tTEST_ACC: " << test_acc << std::endl;
        model.clear();
    }
}
