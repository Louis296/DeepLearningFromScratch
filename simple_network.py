from common.functions import *
from common.gradient import numerical_gradient
from dataset.mnist import load_mnist
import matplotlib.pyplot as plt


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_grad(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads


if __name__ == '__main__':

    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_loss_list = []
    train_acc_list = []
    test_acc_list = []

    # 设置超参数
    iters_num = 1000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rate = 0.1

    iter_per_epoch = max(train_size / batch_size, 1)

    network = TwoLayerNet(784, 50, 10)

    for i in range(iters_num):
        print("Start!")
        # 获取mini-batch
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        # 计算梯度
        grad = network.numerical_grad(x_batch, t_batch)

        # 更新参数
        for key in ('W1', 'W2', 'b1', 'b2'):
            network.params[key] -= learning_rate * grad[key]

        # 计算更新参数后的loss函数（即学习一次的结果）
        loss = network.loss(x_batch, t_batch)
        train_loss_list.append(loss)

        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("TrainAccuracy:" + str(train_acc) + "TestAccuracy" + str(test_acc))
        if i % iter_per_epoch == 0:
            pass

    # for i in range(len(train_acc_list)):
    #     plt.plot(i,train_acc_list)
