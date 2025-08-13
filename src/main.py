"""."""

import numpy as np
import itertools

I_hat = np.eye(4, dtype=np.int32)
J = np.full((4, 4), 1, dtype=np.int32)

A = np.block([
    [J, I_hat, I_hat, I_hat],
    [I_hat, J, I_hat, I_hat],
    [I_hat, I_hat, J, I_hat],
    [I_hat, I_hat, I_hat, J]])

G = np.block([np.eye(16, dtype=np.int32), A])

# H = np.block([-A.T, np.eye(16)])
H = np.block([A.T, np.eye(16)])


for tp_ind in itertools.combinations(range(32), 2):
    e = np.zeros(32)
    for i in tp_ind:
        e[i] = 1
    s = H @ e.T
