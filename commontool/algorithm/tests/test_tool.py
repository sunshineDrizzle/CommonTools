import numpy as np

from commontool.algorithm.tool import calc_overlap


def test_calc_overlap():
    c1 = [1, 2, 3, 4]
    c2 = (3, 4, 5, 6)
    print(calc_overlap(c1, c2))

    array1 = np.array([1, 2, 3, 2])
    array2 = np.array([4, 5, 6, 4])
    print(calc_overlap(array1, array2, 2, 4))
