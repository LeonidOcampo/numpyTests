import numpy as np
from numba import jit
import time


@jit(nopython=True)
def squared_array(array):
    array2 = np.copy(array)
    for i, element in enumerate(array):
        array2[i] = element**2
    return array2


if __name__ == "__main__":
    x = np.linspace(1, 10, 10)
    print(x)
    t1 = time.time()
    y1 = squared_array(x)
    t2 = time.time()
    y2 = squared_array(x)
    t3 = time.time()
    y3 = np.power(x, 2)
    t4 = time.time()
    print("compilation time: %s" % (t2 - t1))
    print(y1)
    print("first numba evaluation: %s" % (t3 - t2))
    print(y2)
    print("numpy time: %s" % (t4 - t3))
    print(y3)
