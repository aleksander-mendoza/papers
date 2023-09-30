import os

import ecc_py
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

def open_img(path: str) -> np.ndarray:
    img = Image.open(path)
    img = np.array(img)
    return img


class RandPatch:

    def __init__(self, height, width, channels):
        self.patch_dims = np.array([height, width, channels])
        self.n = self.patch_dims.prod()

    def __call__(self, img: np.ndarray):
        y, x = np.random.randint((0, 0), img.shape[:2] - self.patch_dims[:2])
        patch = img[y:y + self.patch_dims[0], x:x + self.patch_dims[1]]
        if patch.dtype == np.uint8:
            patch = patch / np.float32(255)
        if self.patch_dims[2] == 1 and patch.ndim == 3 and patch.shape[2] == 3:
            patch = patch @ np.float32([0.2989, 0.5870, 0.1140])  # convert to grayscale first
        if patch.ndim == 3:
            assert patch.shape[2] == self.patch_dims[2]
        assert np.all(patch.shape[:2] == self.patch_dims[:2]), f"{patch.shape}!={self.patch_dims} for img of shape {img.shape}"
        if self.patch_dims[2] > 1:
            assert patch.shape[2] == self.patch_dims[2]
        return patch


class SampleOfCardinality:

    def __init__(self, cardinality:int):
        self.cardinality = cardinality

    def __call__(self, x: np.ndarray):
        if x.dtype == np.uint8:
            x = x / np.float32(255.)
        x = ecc_py.sample_of_cardinality(x, self.cardinality)
        return x


class SampleByThreshold:

    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, x:np.ndarray):
        if x.dtype == np.uint8:
            x = x / np.float32(255.)
        x = x > self.threshold
        return x


class DatasetDir:

    def __init__(self, img_dir="../data/imgs"):
        if not os.path.exists(img_dir):
            print("Put whatever images you like in " + os.path.abspath(img_dir))
            exit()
        self.imgs = os.listdir(img_dir)
        self.dir = img_dir

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return open_img(os.path.join(self.dir,self.imgs[i]))


class Trainer:

    def __init__(self, ecc_net, dataset, rand_patch: RandPatch, preprocess = None):
        self.rand_patch = rand_patch
        self.preprocess = preprocess
        self.dataset = dataset
        self.ecc_net = ecc_net

    def rand_img(self):
        idx = np.random.randint(len(self.dataset))
        img = self.dataset[idx]
        return img

    def train(self, num_images: int, num_patches_per_img: int):
        for _ in tqdm(range(num_images), desc="Training"):
            img = self.rand_img()
            for _ in range(num_patches_per_img):
                x = self.rand_patch(img)
                if self.preprocess is not None:
                    x = self.preprocess(x)
                self.ecc_net(x, learn=True)

    def eval(self, num_images: int, num_patches_per_img: int):
        m = self.ecc_net.m
        shape = list(self.rand_patch.patch_dims)
        if shape[2] == 1:
            shape = shape[:2]
        shape.append(m)
        sums = np.zeros(shape)
        counts = np.zeros(m)
        for _ in tqdm(range(num_images), desc="Evaluating"):
            img = self.rand_img()
            for _ in range(num_patches_per_img):
                x = self.rand_patch(img)
                if self.preprocess is not None:
                    x = self.preprocess(x)
                y = self.ecc_net(x, learn=False)
                sums[..., y] += x
                counts[y] += 1
        means = sums / counts
        return means, counts


class Plot:

    def __init__(self, rows, cols):
        self.fig, self.axs = plt.subplots(rows, cols)
        self.cols, self.rows = cols, rows
        self.m = rows * cols

    def plot(self, means):
        for col in range(self.cols):
            for row in range(self.rows):
                # weights_img = W[:, col + row * cols].reshape(PATCH_H, PATCH_W, PATCH_C)
                self.axs[row, col].imshow(means[..., col + row * self.cols])
        plt.pause(0.1)

