import ecc_py
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# size of environment
r = 8
# transition function
# p:S x A -> S
p = ecc_py.direct_product(ecc_py.cyclic_group(r), ecc_py.cyclic_group(r))
A_LEN = p.shape[1]  # number of generators
s = 0  # current state
m = 4  # size of quotient monoid G/~
n = p.shape[0]  # number of states
C = np.arange(n)
c = C / m
np.random.shuffle(C)


while True:
    for s in p:
        

W = np.random.rand(n, m).astype(np.float32)
W = W / W.sum(0)  # W[h,g] = q(h|g)
U = np.random.rand(m, m, A_LEN).astype(np.float32)
U = U / U.sum(0)  # U[g',g,a] = q(g'|g,a)
g = 0  # current quotient monoid element
#
# preds_u = []
preds_w = []
# gs = []
epsilon = 0.1
fig, (ax_u, ax_w, ax_g) = plt.subplots(3)
unique_indices = []
while True:

    max_w_indices = w.argmax(1)
    for row, max_w_index in enumerate(max_w_indices):
        W[row, max_w_index] += epsilon
    unique_indices.append(len(np.unique(max_w_indices)))
    W = W / W.sum(0)
    preds_w.append(w.max(1).mean())
    ax_w.clear()
    ax_w.plot(preds_w, c='blue')
    ax_g.clear()
    ax_g.plot(unique_indices, c='green')
    plt.pause(0.001)

epsilon = 0.1
while True:
    U_, W_ = ecc_py.learn_uw(p, W)
    # |H| = n
    # |G/~| = m
    # W.shape = (n,m)                 ,  W[h,g]=p(g|h)
    # p.shape = (n,|A|)               ,  p[h,a]=h'
    # assert sum(W[h])=1 for all h
    WA = W[p]  # WA.shape = (n,|A|,m)  ,  WA[h,a,g']=p(g'|ha)
    WA = np.transpose(WA, (1, 0, 2))
    # WA.shape = (|A|,n,m)            ,  WA[a,h,g']=p(g'|ha)
    U = W.T @ WA  # (m,n) @ (|A|,n,m) ,  U[a,g,g']=sum(W[h,g]*WA[a,h,g'] for h in H)
    # U.shape = (|A|,m,m)             ,  U[a,g,g']=p(g'|g,a)
    # assert sum(U[a,g])=1 for all a,g
    Ut = U.transpose(0, 2, 1)
    # Ut.shape = (|A|,m,m)            ,  Ut[a,g',g]=p(g'|g,a)
    UW = Ut @ W.T  # ,  UW[a,g',h]=sum(Ut[a,g',g]*W[h,g] for g in G/~)
    # UW.shape = (|A|,m,n)            ,  UW[a,g',h]=p(g'|h,a)
    UWt = UW.transpose(2, 0, 1)
    # UWt.shape = (n,|A|,m)           ,  UWt[h,a,g']=p(g'|h,a)

    W = (W.T / W.sum(1)).T
    U = (U.T / U.sum(2).T).T

while True:
    U = (U.T / U.sum(2).T).T
    W = (W.T / W.sum(1)).T
    a = np.random.randint(0, A_LEN)  # next action
    g_pred_u = U[a, g]  # recurrent prediction of new monoid element
    s = p[s, a]  # new state
    g_pred_w = W[s]  # feedforward prediction of new monoid element
    g_pred = g_pred_u * g_pred_w
    g = np.argmax(g_pred)
    preds_u.append(g_pred_u[g])
    preds_w.append(g_pred_w[g])
    gs.append(g / m)
    g_pred_u[g] += epsilon
    g_pred_w[g] += epsilon
    ax_u.clear()
    ax_u.plot(preds_u, c='red')
    ax_w.clear()
    ax_w.plot(preds_w, c='blue')
    ax_g.clear()
    ax_g.plot(gs, c='green')
    plt.pause(0.001)

print()