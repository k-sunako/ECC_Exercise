"""."""

import numpy as np


I_hat = np.eye(4, dtype=np.int32)
J = np.full((4, 4), 1, dtype=np.int32)

A = np.block([
    [J, I_hat, I_hat, I_hat],
    [I_hat, J, I_hat, I_hat],
    [I_hat, I_hat, J, I_hat],
    [I_hat, I_hat, I_hat, J]])

G = np.block([np.eye(16, dtype=np.int32), A])
