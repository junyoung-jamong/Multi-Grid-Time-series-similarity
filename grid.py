import numpy as np
import random

class Grid:
    def __init__(self, m=None, n=None):
        if m is not None and n is not None:
            self.m = m
            self.n = n
        else:
            self.m = random.randint(5, 40)
            self.n = random.randint(5, 35)

    def dataset2Matrices(self, ts_set):
        matrices = []
        for ts in ts_set:
            matrices.append(self.ts2Matrix(ts))

        return np.array(matrices)

    def ts2Matrix(self, ts):
        matrix = np.zeros((self.m, self.n))
        T = len(ts)

        height = 1.0/self.m
        width = T/self.n

        for idx in range(T):
            i = int((1-ts[idx])/height)
            if i == self.m:
                i -= 1

            t = idx+1
            j = t/width
            if int(j) == round(j, 7):
                j = int(j)-1
            else:
                j = int(j)

            matrix[i][j] += 1
        return matrix