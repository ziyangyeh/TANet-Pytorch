import numpy as np

def rearrange(npary: np.ndarray)->np.ndarray:
    """
    Rearrange the oder of the label of the input numpy array.
    :param npary: Input numpy array.
    :return:
    """
    npary[npary == 11] = 7
    npary[npary == 12] = 6
    npary[npary == 13] = 5
    npary[npary == 14] = 4
    npary[npary == 15] = 3
    npary[npary == 16] = 2
    npary[npary == 17] = 1
    npary[npary == 21] = 8
    npary[npary == 22] = 9
    npary[npary == 23] = 10
    npary[npary == 24] = 11
    npary[npary == 25] = 12
    npary[npary == 26] = 13
    npary[npary == 27] = 14
    npary[npary == 31] = 8+14
    npary[npary == 32] = 9+14
    npary[npary == 33] = 10+14
    npary[npary == 34] = 11+14
    npary[npary == 35] = 12+14
    npary[npary == 36] = 13+14
    npary[npary == 37] = 14+14
    npary[npary == 41] = 7+14
    npary[npary == 42] = 6+14
    npary[npary == 43] = 5+14
    npary[npary == 44] = 4+14
    npary[npary == 45] = 3+14
    npary[npary == 46] = 2+14
    npary[npary == 47] = 1+14
    return npary