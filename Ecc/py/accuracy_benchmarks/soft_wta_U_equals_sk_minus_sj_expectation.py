import ecc_py
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np

MNIST = torchvision.datasets.MNIST('../../data/', train=False, download=True)
MNIST = MNIST.data.numpy()
TRAINSET_SIZE = 1000
TESTSET_SIZE = 1000
MNIST = MNIST / 255 > 0.8
TRAIN_SET = MNIST[:TRAINSET_SIZE]
TEST_SET = MNIST[TRAINSET_SIZE:TRAINSET_SIZE + TESTSET_SIZE]
# TEST_SET_INDICES, TEST_SET_OFFSETS = ecc_py.batch_dense_to_sparse(TEST_SET)
# TRAIN_SET_INDICES, TRAIN_SET_OFFSETS = ecc_py.batch_dense_to_sparse(TRAIN_SET)
INPUT_SHAPE = (28, 28, 1)
KERNELS = [5, 5, 5, 5, 5, 5, 5]
CHANNELS = [16, 32, 64, 64, 64, 64]
CONV_SHAPES = []
WEIGHTS = []
LATERAL = []


def init_network(input_shape):
    output_shape = input_shape
    for k, c in zip(KERNELS, CHANNELS):
        conv_shape = ecc_py.ConvShape.new_in(output_shape, c, (k, k), (1, 1))
        CONV_SHAPES.append(conv_shape)
        output_shape = conv_shape.out_shape
        W = np.float32(np.random.rand(*conv_shape.minicolumn_w_shape))
        W = W / W.sum(0)
        WEIGHTS.append(W)
        U = np.float32(np.random.rand(*conv_shape.minicolumn_u_shape))
        LATERAL.append(U)
    return output_shape


OUTPUT_SHAPE = init_network(INPUT_SHAPE)
W_epsilon = 0.0001
U_epsilon = 0.001
threshold = 0.1


def run(x, layer, learn):
    U = LATERAL[layer]
    W = WEIGHTS[layer]
    # W.shape == (kernel_height, kernel_width, in_channels, out_channels)
    # U.shape == (out_channels, out_channels)
    conv_shape = CONV_SHAPES[layer]
    x = ecc_py.dense_to_sparse(x.reshape(-1))
    s = conv_shape.sparse_dot_repeated(x, W)
    y = (s > threshold).view(np.ubyte) * 2
    ecc_py.soft_wta_u_repeated_conv_(U, s, y)
    y = y.view(bool)
    if learn:
        y_sparse = ecc_py.dense_to_sparse(y)
        conv_shape.sparse_increment_repeated(W, W_epsilon, x, y_sparse, biased=True)
        conv_shape.update_u_as_expected_sk_minus_sj_repeated(W, W_epsilon, s, y_sparse, U)
    return y


def train():
    for layer in range(len(KERNELS)):
        shape = CONV_SHAPES[layer]
        for _ in range(1000):
            x = rand_patch(TRAIN_SET, shape.kernel)
            run(x, layer, True)
        for _ in range(1000):
            x = rand_patch(TEST_SET, shape.kernel)
            y = run(x, layer, False)


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
            print(probability / probability.sum())

            for i in range(w):
                for j in range(h):
                    s = stats[i + j * w].reshape(PATCH_SIZE)
                    axs[i, j].imshow(s)
            plt.pause(0.01)


train()
