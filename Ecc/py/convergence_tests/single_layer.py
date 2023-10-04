import numpy as np


class HardWtaL2:

    def __init__(self, n, m, l=2, w_step=0.0001, r_step=1. / 1024 * 2):
        self.r_step = r_step
        self.W = np.random.rand(n, m)
        self.r = np.zeros(m)
        self.w_step = w_step
        self.n = n
        self.m = m
        if l == 1:
            self.norm = np.sum
        elif l == 2:
            self.norm = np.linalg.norm
        else:
            raise Exception("Unimplemented norm l" + str(l))
        self.W /= self.norm(self.W, axis=0)

    def __call__(self, x, learn=False):
        x = x.reshape(-1)
        k = np.argmax(x @ self.W + self.r)
        if learn:
            self.r[k] -= self.r_step
            self.W[x, k] += self.w_step / x.sum()
            self.W[:, k] /= self.norm(self.W[:, k])
        return k


class HardWtaZeroOrder(HardWtaL2):

    def __init__(self, n, m, w_step=0.0001, r_step=1. / 1024 * 2):
        super().__init__(n, m, l=1, w_step=w_step, r_step=r_step)


class HardWtaL1:

    def __init__(self, n, m, a=None, Q_step=0.0001, r_step=1. / 1024 * 2):
        self.Q_step = Q_step
        self.r_step = r_step
        self.Q = np.random.rand(n, m)
        self.r = np.zeros(m)
        self.n = n
        self.m = m
        self.a = n // 5 if a is None else a
        self.W = self.Q.argpartition(-self.a, axis=0)[-self.a:]  # encode a sparse binary matrix as a list of indices

    def __call__(self, x, learn=False):
        x = x.reshape(-1)
        k = np.argmax(x.take(self.W).sum(0) + self.r)
        if learn:
            self.r[k] -= self.r_step
            self.Q[:, k] = (1. - self.Q_step) * self.Q[:, k] + self.Q_step * x
            self.W = self.Q.argpartition(-self.a, axis=0)[-self.a:]
        return k


class SoftWta:

    @staticmethod
    def from_formula(n, m, formula, norm=2, W_step=0.0001, U_step=0.001, threshold=0.1):
        if formula == "U=E[s_k-s_j]":
            return SoftWta(n, m, norm=norm, W_step=W_step, U_step=U_step, threshold=threshold,
                           cos_sim=False, use_abs=False, conditional=False)
        elif formula == "U=E[s_k-s_j|s_k>s_j]":
            return SoftWta(n, m, norm=norm, W_step=W_step, U_step=U_step, threshold=threshold,
                           cos_sim=False, use_abs=False, conditional=True)
        elif formula == "U=E[|s_k-s_j|]":
            return SoftWta(n, m, norm=norm, W_step=W_step, U_step=U_step, threshold=threshold,
                           cos_sim=False, use_abs=True, conditional=False)
        elif formula == "U=E[|s_k-s_j||s_k>s_j]":
            return SoftWta(n, m, norm=norm, W_step=W_step, U_step=U_step, threshold=threshold,
                           cos_sim=False, use_abs=True, conditional=True)
        elif formula == "U=E[s_k*s_j]":
            return SoftWta(n, m, norm=norm, W_step=W_step, U_step=U_step, threshold=threshold,
                           cos_sim=True, use_abs=False, conditional=False)
        elif formula == "U=E[s_k*s_j|s_k>s_j]":
            return SoftWta(n, m, norm=norm, W_step=W_step, U_step=U_step, threshold=threshold,
                           cos_sim=True, use_abs=False, conditional=True)
        else:
            raise Exception("Unknown formula " + formula)

    def __init__(self, n, m, norm=2, W_step=0.0001, U_step=0.001, threshold=0.1, cos_sim=False, use_abs=True,
                 conditional=True):
        self.m = m
        self.n = n
        self.cos_sim = cos_sim
        self.conditional = conditional
        self.use_abs = use_abs
        self.threshold = threshold
        self.U_step = U_step
        self.W_step = W_step
        W = np.float32(np.random.rand(n, m))
        if norm == 2:
            self.norm = np.linalg.norm
        elif norm == 1:
            self.norm = np.sum
        else:
            raise Exception("Unknown norm " + str(norm))
        self.W = W / self.norm(W, axis=0)
        # U is row-major. Element U[k,j]==0 means neuron k (row) can inhibit neuron j (column).
        self.U = np.float32(np.random.rand(m, m))

    def __call__(self, x, learn=False):
        import ecc_py
        s = x.reshape(-1) @ self.W
        y = (s > self.threshold).view(np.ubyte) * 2
        # The vector y can either hold 0, 1 or 2:
        # 0 = neuron is shunned (cannot fire),
        # 1 = neuron has fired,
        # 2 = not determined yet and soft_wta_u_ will compute it
        # Therefore we set all neurons below the threshold to 0 so that we rule them out. The remaining neurons
        # are set to 2 co now they will compete with each other as usual according to rules of Soft-WTA.
        ecc_py.swta_u_(self.U, s, y)
        y = y.view(bool)
        if learn:
            self.W[np.ix_(x, y)] += self.W_step
            self.W = self.W / self.norm(self.W, axis=0)
            sk = np.tile(s[y], (self.m, 1)).T
            sk_minus_sj = sk - s
            # Either uses cosine similarity s_k * s_j or differential s_k - s_j
            u_update = np.outer(s, s) if self.cos_sim else sk_minus_sj
            # Differential can be turned into euclidean distance
            u_update = abs(sk_minus_sj) if self.use_abs else u_update
            if self.conditional:  # U = E[u_update | s_k > s_j]
                mask = sk_minus_sj > 0
                self.U[y] *= 1 - mask * self.U_step
                self.U[y] += mask * u_update * self.U_step
            else:  # U = E[u_update]
                self.U[y] *= (1 - self.U_step)
                self.U[y] += (u_update * self.U_step)
        return y


def from_tag(tag, m, n, weight_step=0.0001, inhibition_step=0.001):
    if tag == "hwtaL1":
        return HardWtaL1(n=n, m=m, Q_step=weight_step, r_step=inhibition_step)
    elif tag == "hwtaL2":
        return HardWtaL2(n=n, m=m, w_step=weight_step, r_step=inhibition_step)
    elif tag == "swtaL1":
        return SoftWta(n=n, m=m, norm=1, W_step=weight_step, U_step=inhibition_step)
    elif tag == "swtaL2":
        return SoftWta(n=n, m=m, norm=2, W_step=weight_step, U_step=inhibition_step)
    elif tag == "hwta0":
        return HardWtaZeroOrder(n=n, m=m, w_step=weight_step, r_step=inhibition_step)
