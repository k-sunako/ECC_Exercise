"""."""

import numpy as np
import itertools

n = 32
k = 16

I_hat = np.eye(4, dtype=np.int32)
J = np.full((4, 4), 1, dtype=np.int32)

A = np.block([
    [J, I_hat, I_hat, I_hat],
    [I_hat, J, I_hat, I_hat],
    [I_hat, I_hat, J, I_hat],
    [I_hat, I_hat, I_hat, J]])

G = np.block([np.eye(k, dtype=np.int32), A])

H = np.block([A.T, np.eye(k, dtype=np.int32)])

syns = {}

syns[tuple([0]*k)] = np.zeros(n, dtype=np.int32)

for tp_ind in itertools.combinations(range(n), 1):
    e = np.zeros(n, dtype=np.int32)
    for i in tp_ind:
        e[i] = 1
    s = (H @ e.T) % 2
    syns[tuple(s.tolist())] = e

for tp_ind in itertools.combinations(range(n), 2):
    e = np.zeros(n, dtype=np.int32)
    for i in tp_ind:
        e[i] = 1
    s = (H @ e.T) % 2
    syns[tuple(s.tolist())] = e

for tp_ind in itertools.combinations(range(n), 3):
    e = np.zeros(n, dtype=np.int32)
    for i in tp_ind:
        e[i] = 1
    s = (H @ e.T) % 2
    syns[tuple(s.tolist())] = e

N = 100000
n_cant_decode = 0
p = 0.01
for _ in range(N):
    u = np.random.randint(0, 2, k, dtype=np.int32)
    c = (u @ G) % 2
    np.testing.assert_array_equal((H @ c.T) % 2, np.zeros(k, dtype=np.int32))
    e = np.where(np.random.uniform(0.0, 1.0, n) < p, 1, 0)
    r = c + e
    s = (H @ r.T) % 2
    if tuple(s.tolist()) not in syns:
        n_cant_decode += 1
print(n_cant_decode / N)
