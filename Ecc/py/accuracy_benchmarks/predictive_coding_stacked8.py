import math
import re
from builtins import input

import ecc_py
import os
from matplotlib import pyplot as plt
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision

fig, axs = None, None

SAMPLES = 60000
DIR = 'predictive_coding_stacked8/'
SPLITS = [20, 100, 1000, 0.1]
SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

def closest_divisors(n):
    a = round(math.sqrt(n))
    while n%a > 0: a -= 1
    return a,n//a

def visualise_connection_heatmap(in_w, in_h, ecc_net, out_w, out_h, pause=None):
    global fig, axs
    if fig is None:
        fig, axs = plt.subplots(out_w, out_h)
        for a in axs:
            for b in a:
                b.set_axis_off()
    for i in range(out_w):
        for j in range(out_h):
            w = ecc_net.get_weights(i + j * out_w)
            w = np.array(w)
            w = w.reshape(in_w, in_h)
            w.strides = (8, 56)
            axs[i, j].imshow(w)
    if pause is None:
        plt.show()
    else:
        plt.pause(pause)


def visualise_recursive_weights(w, h, ecc_net):
    fig, axs = plt.subplots(w, h)
    for a in axs:
        for b in a:
            b.set_axis_off()
    for i in range(w):
        for j in range(h):
            weights = np.array(ecc_net.get_weights(i + j * w))
            weights[i + j * w] = 0
            weights = weights.reshape([w, h]).T
            axs[i, j].imshow(weights)
    plt.show()


def compute_confusion_matrix_fit(conf_mat, ecc_net, metric_l2=True):
    assert ecc_net.out_grid == [1, 1]
    fit = torch.empty(ecc_net.out_channels)
    for i in range(ecc_net.out_channels):
        corr = conf_mat[i] / conf_mat[i].sum()
        wei = torch.tensor(ecc_net.get_weights(i))
        if metric_l2:
            fit[i] = corr @ wei
        else:  # l1
            fit[i] = (corr - wei).abs().sum()
    return fit


class MachineShape:

    def __init__(self, layer_type,
                 input_size: (int, int),
                 channels: [int],
                 kernels: [(int, int)],
                 strides: [(int, int)],
                 drifts: [int]):
        assert len(channels) == len(kernels) + 1
        assert len(kernels) == len(strides) == len(drifts)
        self.layer_type = layer_type
        if self.layer_type == ecc_py.HwtaL2Layer:
            self.layer_code = 'h2'
        elif self.layer_type == ecc_py.SwtaLayer:
            self.layer_code = 's'
        else:
            raise ValueError("Invalid layer type " + str(layer_type))

        self.channels = channels
        self.kernels = kernels
        self.strides = strides
        self.drifts = drifts
        self.h, self.w = input_size
        input_size = self.input_shape = (self.h, self.w, self.channels[0])
        self.shapes:[ecc_py.ConvShape] = []
        for k, co, s in zip(self.kernels, self.channels[1:], self.strides):
            sh = ecc_py.ConvShape.new_in(input_size, out_channels=co, kernel=(k, k), stride=(s, s))
            self.shapes.append(sh)
            input_size = sh.out_shape
        self.output_shape = input_size

    def composed_conv(self, idx)->ecc_py.ConvShape:
        strides = [[s, s] for s in self.strides[:idx+1]]
        kernels = [[k, k] for k in self.kernels[:idx+1]]
        stride, kernel = ecc_py.conv_compose_array(strides=strides, kernels=kernels)
        c = self.channels[idx+1]
        return ecc_py.ConvShape.new_in(self.input_shape, out_channels=c, kernel=kernel[:2], stride=stride[:2])

    def code_name_part(self, idx):
        k = "k" + str(self.kernels[idx])
        s = "s" + str(self.strides[idx])
        c = "c" + str(self.channels[idx])
        d = "d" + str(self.drifts[idx])
        return k + s + c + d

    def code_name(self, idx):
        if idx <= 0:
            return 'mnist'
        path = ''.join([self.code_name_part(i) + "_" for i in range(idx)])
        return self.layer_code + '_' + path + "c" + str(self.channels[idx])

    def parent_code_name(self):
        if len(self) == 0:
            return None
        else:
            return self.code_name(len(self) - 1)

    def save_file(self, idx, dest_dir="results"):
        return os.path.join(dest_dir, self.code_name(idx))

    def kernel(self, idx):
        return [self.kernels[idx], self.kernels[idx]]

    def stride(self, idx):
        return [self.strides[idx], self.strides[idx]]

    def drift(self, idx):
        return [self.drifts[idx], self.drifts[idx]]

    def __len__(self):
        return len(self.kernels)

    def load_layer(self, idx, dest_dir="results"):
        mf = self.save_file(idx + 1, dest_dir=dest_dir) + ".model"
        if os.path.exists(mf):
            return self.layer_type.load(mf)
        else:
            return None

    def load_or_save_params(self, idx, dest_dir="results", **kwrds):
        f = self.save_file(idx + 1, dest_dir=dest_dir) + " params.txt"
        if os.path.exists(f):
            with open(f, "r") as f:
                d2 = json.load(f)
                kwrds.update(d2)
        else:
            with open(f, "w+") as f:
                json.dump(kwrds, f)
        return kwrds


class Dataset:

    def __init__(self, machine_shape: MachineShape, layer_idx):
        self.machine_shape = machine_shape
        self.layer_idx = layer_idx
        self.file = machine_shape.save_file(layer_idx) + " data.npz"
        self.composed_stride, self.composed_kernel = machine_shape.composed_conv(layer_idx)
        load_file = machine_shape.save_file(layer_idx - 1) + " data.npz"
        if not os.path.exists(load_file) and layer_idx == 0:
            mnist = torchvision.datasets.MNIST('../../data/', train=False, download=True)
            self.indices, self.offsets = ecc_py.batch_sparse(mnist)
            np.save(load_file, (self.indices, self.offsets))
        else:
            self.indices, self.offsets = np.load(load_file)

    def __getitem__(self, index):
        return self.indices[self.offsets[index]:self.offsets[index+1]]

    def __len__(self):
        return len(self.offsets)-1

    def save_mnist(self):
        np.save(self.file, (self.indices, self.offsets))


class Net:

    def __init__(self, machine_shape: MachineShape):
        self.machine_shape = machine_shape
        self.layer: ecc_py.SwtaLayer = self.machine_shape.load_layer(len(machine_shape) - 1)

    def train(self, plot=False, save=True,
              snapshots_per_sample=1,
              iterations=2000000,
              interval=100000,
              test_patches=20000):
        idx = len(self.machine_shape) - 1
        drift = self.machine_shape.drift(idx)
        params = self.machine_shape.load_or_save_params(
            idx,
            input_shape=self.machine_shape.input_shape,
            snapshots_per_sample=snapshots_per_sample,
            iterations=iterations,
            interval=interval,
            test_patches=test_patches,
        )
        snapshots_per_sample = params['snapshots_per_sample']
        iterations = params['iterations']
        interval = params['interval']
        test_patches = params['test_patches']
        input_dataset = Dataset(self.machine_shape, idx)
        mnist = Dataset(self.machine_shape, 0)
        composed_shape = self.machine_shape.composed_conv(idx)
        layer_shape = self.machine_shape.shapes[idx]
        test_patch_output_positions = np.random.randint((0,0), composed_shape.out_grid, (test_patches,2))
        test_input_patches = []
        test_img_patches = []
        for test_patch_output_position in test_patch_output_positions:
            rand_img_idx = np.random.randint(0, len(mnist))
            mnist_img = mnist[rand_img_idx]
            input_img = input_dataset[rand_img_idx]
            mnist_patch = composed_shape.sparse_kernel_column_input_subset_reindexed(mnist_img, tuple(test_patch_output_position))
            input_patch = layer_shape.sparse_kernel_column_input_subset_reindexed(input_img, tuple(test_patch_output_position))
            test_input_patches.append(input_patch)
            test_img_patches.append(mnist_patch)

        print("PATCH_SIZE=", composed_shape.kernel_column_shape)
        all_missed = []
        all_total_sum = []
        if plot:
            w, h = closest_divisors(layer_shape.out_channels)
            fig, axs = plt.subplots(w, h)
            for a in axs:
                for b in a:
                    b.set_axis_off()

        def eval_ecc():
            test_outputs, s_sums, missed = test_input_patches.batch_infer_and_measure_s_expectation(layer)
            receptive_fields = test_img_patches.measure_receptive_fields(test_outputs)
            receptive_fields = receptive_fields.squeeze(2)
            all_missed.append(missed / test_patches)
            all_total_sum.append(s_sums)
            print("missed=", all_missed)
            print("sums=", all_total_sum)
            if plot:
                for i in range(w):
                    for j in range(h):
                        axs[i, j].imshow(receptive_fields[:, :, i + j * w])
                plt.pause(0.01)
                if save:
                    img_file_name = self.machine_shape.save_file(idx + 1) + " before.png"
                    if s > 0 or os.path.exists(img_file_name):
                        img_file_name = self.machine_shape.save_file(idx + 1) + " after.png"
                    plt.savefig(img_file_name)

        for s in tqdm(range(int(math.ceil(iterations / interval))), desc="training"):
            # eval_ecc()
            for _ in range(interval):
                rand_idx = np.random.randint(0, len(input_dataset))
                x = rand_idx[rand_idx]
                out_positions = np.random.randint((0, 0), composed_shape.out_grid, (snapshots_per_sample, 2))
                for out_pos in range(out_positions):
                    input_patch = layer_shape.sparse_kernel_column_input_subset_reindexed(x, tuple(out_pos))
                    y = self.layer.train(input_patch)
            if save:
                self.layer.save(self.machine_shape.save_file(idx + 1) + ".model")
        # eval_ecc()
        with open(self.machine_shape.save_file(idx + 1) + ".log", "a+") as f:
            print("missed=", all_missed, file=f)
            print("sums=", all_total_sum, file=f)
        if plot:
            plt.close(fig)
            # plt.show()

    def eval_with_classifier_head(self, overwrite_data=False, overwrite_benchmarks=False, epochs=4):
        idx = len(self.machine_shape) - 1
        print("PATCH_SIZE=", self.m.in_shape)
        layer = self.m.get_layer(idx)
        benchmarks_save = self.machine_shape.save_file(idx + 1) + " accuracy2.txt"
        out_mnist = Mnist(self.machine_shape, idx + 1)
        if os.path.exists(out_mnist.file) and not overwrite_data:
            out_mnist.load()
        else:
            in_mnist = Mnist(self.machine_shape, idx)
            in_mnist.load()
            out_mnist.mnist = in_mnist.mnist.batch_infer(layer)
            out_mnist.save_mnist()
        if not os.path.exists(benchmarks_save) or overwrite_benchmarks:

            class D(torch.utils.data.Dataset):
                def __init__(self, imgs, lbls):
                    self.imgs = imgs
                    self.lbls = lbls

                def __len__(self):
                    return len(self.imgs)

                def __getitem__(self, idx):
                    return self.imgs.to_f32_numpy(idx), self.lbls[idx]

            for split in SPLITS:  # [0.1, 0.2, 0.5, 0.8, 0.9]:
                if type(split) is float:
                    train_len = int(len(MNIST) * split)
                else:
                    train_len = split
                eval_len = len(MNIST) - train_len
                train_data = out_mnist.mnist.subdataset(0, train_len)
                eval_data = out_mnist.mnist.subdataset(train_len)
                train_lbls = LABELS[0:train_len].numpy()
                eval_lbls = LABELS[train_len:].numpy()
                train_d = D(train_data, train_lbls)
                eval_d = D(eval_data, eval_lbls)
                linear = torch.nn.Linear(out_mnist.mnist.volume, 10)
                loss = torch.nn.NLLLoss()
                optim = torch.optim.Adam(linear.parameters())
                bs = 64
                train_dataloader = DataLoader(train_d, batch_size=bs, shuffle=True)
                eval_dataloader = DataLoader(eval_d, batch_size=bs, shuffle=True)

                for epoch in range(epochs):
                    train_accuracy = 0
                    train_total = 0
                    for x, y in tqdm(train_dataloader, desc="train"):
                        optim.zero_grad()
                        bs = x.shape[0]
                        x = x.reshape(bs, -1)
                        x = linear(x)
                        x = torch.log_softmax(x, dim=1)
                        d = loss(x, y)
                        train_accuracy += (x.argmax(1) == y).sum().item()
                        train_total += x.shape[0]
                        d.backward()
                        optim.step()

                    eval_accuracy = 0
                    eval_total = 0
                    for x, y in tqdm(eval_dataloader, desc="eval"):
                        bs = x.shape[0]
                        x = x.reshape(bs, -1)
                        x = linear(x)
                        eval_accuracy += (x.argmax(1) == y).sum().item()
                        eval_total += x.shape[0]
                    s = "split=" + str(split) + \
                        ", train_len=" + str(train_len) + \
                        ", eval_len=" + str(eval_len) + \
                        ", epoch=" + str(epoch) + \
                        ", train_accuracy=" + str(train_accuracy / train_total) + \
                        ", eval_accuracy=" + str(eval_accuracy / eval_total)
                    with open(benchmarks_save, 'a+') as f:
                        print(s, file=f)
                    print(s)

    def eval_with_naive_bayes(self, overwrite_data=False, overwrite_benchmarks=False, min_deviation_from_mean=None):
        idx = len(self.machine_shape) - 1
        print("PATCH_SIZE=", self.m.in_shape)
        layer = self.m.get_layer(idx)
        i = "I" if min_deviation_from_mean is not None else ""
        benchmarks_save = self.machine_shape.save_file(idx + 1) + " accuracy" + i + ".txt"
        out_mnist = Mnist(self.machine_shape, idx + 1)
        if os.path.exists(out_mnist.file) and not overwrite_data:
            out_mnist.load()
        else:
            in_mnist = Mnist(self.machine_shape, idx)
            in_mnist.load()
            out_mnist.mnist = in_mnist.mnist.batch_infer(layer)
            out_mnist.save_mnist()
        if not os.path.exists(benchmarks_save) or overwrite_benchmarks:
            with open(benchmarks_save, 'a+') as f:
                for split in SPLITS:
                    train_len = int(len(MNIST) * split) if type(split) is float else split
                    eval_len = len(MNIST) - train_len
                    train_data = out_mnist.mnist.subdataset(0, train_len)
                    eval_data = out_mnist.mnist.subdataset(train_len)
                    train_lbls = LABELS[0:train_len].numpy()
                    eval_lbls = LABELS[train_len:].numpy()
                    lc = train_data.fit_naive_bayes(train_lbls, 10,
                                                    invariant_to_column=min_deviation_from_mean is not None)
                    lc.clear_class_prob()
                    train_out_lbl = lc.batch_classify(train_data, min_deviation_from_mean)
                    eval_out_lbl = lc.batch_classify(eval_data, min_deviation_from_mean)
                    train_accuracy = (train_out_lbl == train_lbls).mean()
                    eval_accuracy = (eval_out_lbl == eval_lbls).mean()
                    s = "split=" + str(split) + \
                        ", train_len=" + str(train_len) + \
                        ", eval_len=" + str(eval_len) + \
                        ", train_accuracy=" + str(train_accuracy) + \
                        ", eval_accuracy=" + str(eval_accuracy)
                    print(s, file=f)
                    print(s)


def run_experiments():
    i49 = (49, 6, 1, 1, 1, None, 1)
    e144 = (144, 5, 1, 1, 1, None, 1)
    e200 = (200, 5, 1, 1, 1, None, 1)
    e256 = (256, 5, 1, 1, 1, None, 1)

    def e(channels, kernel, k=1):
        return channels, kernel, 1, 1, 1, None

    def c(channels, drift, k=1):
        return channels, 1, 1, drift, 6, 'in'

    experiments = [
        # (1, [e(49, 6), c(9, 3), e(100, 6), c(9, 5), e(144, 6), c(16, 7), e(256, 6), c(20, 8), e(256, 6)]),
        # (1, [e(49, 6), c(9, 3), e(100, 6, k=4), c(9, 5), e(144, 6, k=3), c(16, 7), e(256, 6, k=4), c(20, 8), e(256, 6, k=4)]),
        # (1, [e(49, 6), e(100, 6,k=4), e(144, 6,k=3), e(256, 6,k=4), e(256, 6,k=4)]),
        # (1, [e(49, 6), e(100, 6), e(144, 6, k=3), e(256, 6, k=4), e(256, 6, k=4)]),
        # (1, [e(49, 6), e(100, 6), e(144, 6), e(256, 6, k=4), e(256, 6, k=4)]),
        # (1, [e(49, 6), e(100, 6), e(144, 6), e(256, 6), e(256, 6, k=4)]),
        #    (1, [e(100, 28)]),
        #    (1, [e(256, 28)]),
        #    (1, [e(400, 28)]),
        #    (1, [e(6, 6), e(6, 6), e(6, 6), e(6, 6), e(6, 6)]),
        #    (1, [e(8, 6), e(2 * 8, 6), e(3 * 8, 6), e(4 * 8, 6), e(4 * 8, 6)]),
        #    (1, [e(8, 6), e(2 * 8, 6), e(3 * 8, 6), e(4 * 8, 6), e(5 * 8, 6)]),
        # (1, [e(8, 6), e(2 * 8, 3), e(3 * 8, 3), e(4 * 8, 3), e(4 * 8, 3), e(4 * 8, 3)]),
        # (1, [e(16, 6), e(2 * 16, 3), e(3 * 16, 3), e(4 * 16, 3), e(4 * 16, 3), e(4 * 16, 3)]),
        # (1, [e(16, 6), e(2 * 16, 6), e(3 * 16, 6), e(4 * 16, 6), e(4 * 16, 6)]),
        (ecc_py.HwtaL2Layer, 1, [e(49, 6), e(100, 6), e(144, 6), e(256, 6), e(256, 6)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c9(7), e144, c9(10), e144, c9(7)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e144, c16(10), e144, c16(7), e144, c16(3)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e200, c16(10), e200, c20(7), e200, c20(3)]),
        # (1, [i49, c9(3), e144, c9(5), e144, c16(7), e200, c20(10), e256, c25(7), e200, c20(3)]),
    ]
    overwrite_benchmarks = False
    overwrite_data = False
    for experiment in experiments:
        layer_type, first_channels, layers = experiment
        kernels, strides, channels, drifts = [], [], [first_channels], []
        for layer in layers:
            channel, kernel, stride, drift, snapshots_per_sample, threshold = layer
            kernels.append(kernel)
            strides.append(stride)
            channels.append(channel)
            drifts.append(drift)
            s = MachineShape(layer_type=layer_type,
                             input_size=[28, 28],
                             channels=channels,
                             kernels=kernels,
                             strides=strides,
                             drifts=drifts)
            code_name = s.save_file(len(kernels))
            save_file = code_name + " data.npz"
            if overwrite_benchmarks or overwrite_data or not os.path.exists(save_file):
                m = Net(s)
                print(save_file)
                m.train(save=True, plot=True, snapshots_per_sample=snapshots_per_sample)
                print(save_file)
                m.eval_with_naive_bayes(overwrite_data=overwrite_data,
                                        overwrite_benchmarks=overwrite_benchmarks)
                print(save_file)
                m.eval_with_classifier_head(overwrite_benchmarks=overwrite_benchmarks)
                print(save_file)
                m.eval_with_naive_bayes(min_deviation_from_mean=0.01,
                                        overwrite_benchmarks=overwrite_benchmarks)
                print(save_file)


def parse_benchmarks(file_name, splits=[0.1, 0.8]):
    if not file_name.startswith('predictive_coding_stacked8/'):
        file_name = DIR + '/' + file_name
    with open(file_name, "r") as f:
        eval_accuracies = [0.] * len(splits)
        train_accuracies = [0.] * len(splits)
        for line in f:
            attributes = line.split(",")
            attributes = [attr.split("=") for attr in attributes]
            attributes = {key.strip(): value for key, value in attributes}
            split_val = float(attributes["split"])
            if split_val in splits:
                i = splits.index(split_val)
                eval_accuracies[i] = max(eval_accuracies[i], float(attributes["eval_accuracy"]))
                train_accuracies[i] = max(train_accuracies[i], float(attributes["train_accuracy"]))
        return eval_accuracies + train_accuracies


EXTRACT_K_S = re.compile("k([0-9]+)s([0-9]+)c([0-9]+)(k[0-9]+)?d([0-9]+)")
HAS_DRIFT = re.compile("d[2-9][0-9]*")
TABLE_MODE = 'csv'  # alternatives: latex, csv
if TABLE_MODE == 'latex':
    TABLE_FIELD_SEP = ' & '
    TABLE_ROW_SEP = ' \\\\\n\\hline\n'
elif TABLE_MODE == 'csv':
    TABLE_FIELD_SEP = ', '
    TABLE_ROW_SEP = '\n'


class ExperimentData:
    def __init__(self, experiment_name: str):
        self.leaf = True
        self.name = experiment_name
        kernels, strides, channels, drifts, ks = [], [], [], [], []
        for m in EXTRACT_K_S.finditer(experiment_name):
            k, s, c, d = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(5))
            kernels.append(k)
            strides.append(s)
            channels.append(c)
            drifts.append(d)
            k = 1 if m.group(4) is None else int(m.group(4)[1:])
            ks.append(k)
        channels.append(int(experiment_name.rsplit('c', 1)[1]))
        self.shape = MachineShape(channels, kernels, strides, drifts, ks)

        self.comp_stride, self.comp_kernel = self.shape.composed_conv(len(self.shape) - 1)
        self.out_shape = htm.conv_out_size([28, 28], self.comp_stride, self.comp_kernel)[:2]
        self.benchmarks = {
            'softmax': parse_benchmarks(self.name + " accuracy2.txt"),
            'vote': parse_benchmarks(self.name + " accuracyI.txt"),
            'naive': parse_benchmarks(self.name + " accuracy.txt"),
        }
        assert min(ks) > 0
        self.has_k = max(ks) > 1
        self.has_drift = HAS_DRIFT.search(experiment_name) is not None

    def acc(self, benchmark, split):
        return "{:.0%}".format(self.benchmark(benchmark, split, False)) + "/" + \
               "{:.0%}".format(self.benchmark(benchmark, split, True))

    def all_acc(self, split):
        return self.acc("softmax", split) + ";" + self.acc("naive", split) + ";" + self.acc("vote", split)

    def benchmark(self, benchmark, split, train):
        benchmark = self.benchmarks[benchmark]
        if split == 1:
            split = 0
        elif split == 8:
            split = 1
        else:
            raise Exception("Should be either 1 or 8")
        return benchmark[(2 if train else 0) + split]

    def format(self, db, detailed=True, metric=True):
        if detailed:
            s = self.shape
            k = str(self.comp_kernel[0]) + "x" + str(self.comp_kernel[1])
            o = str(self.out_shape[0]) + "x" + str(self.out_shape[1])
            if metric:
                codename = [METRIC_STR]
            else:
                codename = ["Yes" if self.has_drift else "No"]
            codename = codename + \
                       ["k" + str(k) + "c" + str(c) + ("/" + str(k_) if k_ > 1 else "") + "d" + str(d) for
                        k, c, d, k_ in
                        zip(s.kernels, s.channels[1:], s.drifts, self.shape.k)]
            acc8 = [k]
            acc1 = [o]

            for i in range(1, len(self.shape)):
                prev_ex = db[self.shape.code_name(i)]
                acc8.append(prev_ex.all_acc(8))
                acc1.append(prev_ex.all_acc(1))
            acc8.append(self.all_acc(8))
            acc1.append(self.all_acc(1))
            assert len(codename) == len(acc8) == len(acc1)
            return TABLE_ROW_SEP.join(
                [TABLE_FIELD_SEP.join(codename), TABLE_FIELD_SEP.join(acc8), TABLE_FIELD_SEP.join(acc1)])
        else:
            k = str(self.comp_kernel[0]) + "x" + str(self.comp_kernel[1])
            o = str(self.out_shape[0]) + "x" + str(self.out_shape[1])
            acc8 = "{:.2f}/{:.2f}".format(self.benchmark("softmax", 8, False), self.benchmark("naive", 8, False))
            acc1 = "{:.2f}/{:.2f}".format(self.benchmark("softmax", 1, False), self.benchmark("naive", 1, False))
            s = self.shape
            prev_softmax8 = [db[self.shape.code_name(i)].benchmark("softmax", 8, False) for i in
                             range(1, len(self.shape))]
            prev_softmax8.append(self.benchmark("softmax", 8, False))
            path = ' '.join(["k" + str(k) + "c" + str(c) + "d" + str(d) + "({:.2f})".format(s8)
                             for k, c, d, s8
                             in zip(s.kernels, s.channels[1:], s.drifts, prev_softmax8)])
            return ', '.join([acc8, acc1, k, o, path])

    def experiment(self, benchmark, overwrite_benchmarks=False):
        if benchmark == "vote":
            save_file = self.name + " accuracyI.txt"
            if not os.path.exists(save_file):
                m = FullColumnMachine(self.shape)
                print(save_file)
                m.eval_with_naive_bayes(min_deviation_from_mean=0.01, overwrite_benchmarks=overwrite_benchmarks)
        elif benchmark == "softmax":
            save_file = self.name + " accuracy2.txt"
            m = FullColumnMachine(self.shape)
            print(save_file)
            m.eval_with_classifier_head(epochs=4, overwrite_benchmarks=overwrite_benchmarks)


class ExperimentDB:

    def __init__(self):
        s = " accuracy2.txt"
        self.experiments = [e[:-len(s)] for e in os.listdir(DIR + '/') if e.endswith(s)]
        self.experiment_data = {n: ExperimentData(n) for n in self.experiments}
        for ex in self.experiment_data.values():
            s: MachineShape = ex.shape
            parent = self.experiment_data.get(s.parent_code_name())
            if parent is not None:
                parent.leaf = False

    def print_accuracy2_results(self, depth, with_drift=None, with_k=None):
        scores = []
        for experiment in self.experiment_data.values():
            if with_drift is not None and with_drift != experiment.has_drift:
                continue
            if type(depth) is list:
                if abs(experiment.comp_kernel[0] - depth[0]) > depth[1]:
                    continue
            elif type(depth) is int:
                if len(experiment.shape) == depth:
                    continue
            if with_k is not None and with_k != experiment.has_k:
                continue
            scores.append(experiment)
        scores.sort(key=lambda x: x.benchmark('softmax', 8, False))
        print("Depth =", depth, ",  with_drift =", with_drift, ",  with_k =", with_k)
        for exp_data in scores:
            print(exp_data.format(self.experiment_data), end=TABLE_ROW_SEP)

    def experiment_on_all(self, mode, overwrite_benchmarks=False):
        for e in self.experiment_data.values():
            e.experiment(mode, overwrite_benchmarks=overwrite_benchmarks)


def print_comparison_across_sample_sizes():
    ss = [20, 100, 1000, 6000, 12000, 60000]
    files = {
        "k6s1c1d1_c49": "k6c49",
        "k6s1c1d1_k6s1c49d1_c100": "k6c100",
        "k6s1c1d1_k6s1c49d1_k6s1c100d1_c144": "k6c144",
        "k6s1c1d1_k6s1c49d1_k6s1c100d1_k6s1c144d1_c256": "k6c256",
        "k6s1c1d1_k6s1c49d1_k6s1c100d1_k6s1c144d1_k6s1c256d1_c256": "k6c256",
    }
    results = {k: {s: [] for s in ss} for k in files.keys()}
    for s in ss:
        d = 'predictive_coding_stacked8/' + str(s)
        suff = " accuracy2.txt"
        for f in files:
            splits = [20, 100, 1000, 0.1, 0.9]
            b = parse_benchmarks(d + '/' + f + suff, splits=splits)
            b = " ".join(["{:.0%}/{:.0%}".format(e, t) for e, t in zip(b[:len(splits)], b[len(splits):])])
            results[f][s] = b
    results = [(k, v) for k, v in results.items()]
    results.sort(key=lambda a: a[0])
    print(TABLE_FIELD_SEP.join(["%"] + [files[k] for k, _ in results]), end=TABLE_ROW_SEP)
    for s in ss:
        print(s, end=TABLE_FIELD_SEP)
        print(TABLE_FIELD_SEP.join([result[s] for _, result in results]), end=TABLE_ROW_SEP)


# print_comparison_across_sample_sizes()

run_experiments()
# edb = ExperimentDB()
# edb.experiment_on_all("softmax", overwrite_benchmarks=True)
# edb.compare_metrics()
# edb.print_accuracy2_results([26, 2], with_drift=False, with_k=False)
# edb.print_accuracy2_results([26, 2], with_drift=False, with_k=False)
