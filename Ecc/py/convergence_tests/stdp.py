
import numpy as np


class STDP:

    def __init__(self, n, m, norm=2, threshold=5, w_step=0.0001, r_step=1. / 1024 * 2):
        self.r_step = r_step
        self.W = np.random.rand(n, m)
        self.r = np.zeros(m)
        self.w_step = w_step
        self.n = n
        self.threshold = threshold
        self.m = m
        self.norm = np.sum if norm == 1 else np.linalg.norm
        self.W /= self.norm(self.W, axis=2)

    def __call__(self, x, learn=False):
        """
        :param x: list of indices. Each index encodes 1 bit in a sparse binary vector. The bits come in a certain order.
        This order is defined by the order of indices in the list.
        :param learn:
        :return: a new list of indices.
        """
        s = np.zeros(self.m)
        for idx in x:
            s += self.W[idx]
            k = s.argmax()
            if s[k] > self.threshold:
                break
        if learn:
            self.r[k] -= self.r_step
            self.W[x, k] += self.w_step / x.sum()
            self.W[:, k] /= self.norm(self.W[:, k])
        return k


if __name__ == '__main__':

    import common as c

    data = c.DatasetDir("../data/imgs")
    plotter = c.Plot(5, 4)
    rand_patch = c.RandPatch(7, 7, 1)
    stdp = STDP(m=plotter.m, n=rand_patch.n, l=1)
    trainer = c.Trainer(stdp, data, rand_patch)

    while True:
        trainer.train(20, 800)
        means, counts = trainer.eval(len(data), 200)
        print("probabilities=", counts / counts.sum())
        plotter.plot(means)