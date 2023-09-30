import ecc_py
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

MNIST = torchvision.datasets.MNIST('../../data/', train=False, download=True)
MNIST = MNIST.data.numpy()
PATCH_SIZE = np.array([5, 5])
w, h = 5, 4
y_len = w*h
W = np.float32(np.random.rand(PATCH_SIZE.prod(), y_len))
W = W / W.sum(0)
U = np.float32(np.random.rand(y_len, y_len))
W_epsilon = 0.0001
U_epsilon = 0.001
threshold = 0.1


def rand_patch():
    r = np.random.rand(2)
    img = MNIST[int(np.random.rand() * len(MNIST))]
    left_bottom = (img.shape - PATCH_SIZE) * r
    left_bottom = left_bottom.astype(int)
    top_right = left_bottom + PATCH_SIZE
    img = img[left_bottom[0]:top_right[0], left_bottom[1]:top_right[1]]
    x = img / 255 > 0.8
    x = x.reshape(-1)
    return x


def run(x, learn):
    global W, U
    s = x @ W
    y = (s > threshold).view(np.ubyte) * 2
    ecc_py.soft_wta_u_(U, s, y)
    y = y.view(bool)
    if learn:
        W[np.ix_(x, y)] += W_epsilon
        W = W / W.sum(0)
        sk = np.tile(s[y], (y_len, 1)).T
        sk_minus_sj = sk - s
        mask = sk_minus_sj > 0
        U[y] *= 1 - mask * U_epsilon
        U[y] += mask * (sk_minus_sj*U_epsilon)
        if sum(s) > 0:
            print(sum(y))
    return y


def experiment():
    fig, axs = plt.subplots(w, h)
    test_patches = [rand_patch() for _ in range(20000)]
    for idx in range(200000):
        x = rand_patch()
        y = run(x, True)
        if idx % 2000 == 0:
            stats = np.zeros((y_len, PATCH_SIZE.prod()))
            probability = np.zeros(y_len)
            for x in test_patches:
                y = run(x, False)
                probability[y] += 1
                stats[y] += x
            print(probability)
            print(probability/probability.sum())
            # print(stats)
            for i in range(w):
                for j in range(h):
                    s = stats[i + j * w].reshape(PATCH_SIZE)
                    axs[i, j].imshow(s)
            plt.pause(0.01)

experiment()