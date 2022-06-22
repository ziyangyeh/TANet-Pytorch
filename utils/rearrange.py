def rearrange(npary):
    """
    Rearrange the oder of the label of the input numpy array.
    :param npary: Input numpy array.
    :return:
    """
    npary[npary == 17] = 1
    npary[npary == 37] = 1
    npary[npary == 16] = 2
    npary[npary == 36] = 2
    npary[npary == 15] = 3
    npary[npary == 35] = 3
    npary[npary == 14] = 4
    npary[npary == 34] = 4
    npary[npary == 13] = 5
    npary[npary == 33] = 5
    npary[npary == 12] = 6
    npary[npary == 32] = 6
    npary[npary == 11] = 7
    npary[npary == 31] = 7
    npary[npary == 21] = 8
    npary[npary == 41] = 8
    npary[npary == 22] = 9
    npary[npary == 42] = 9
    npary[npary == 23] = 10
    npary[npary == 43] = 10
    npary[npary == 24] = 11
    npary[npary == 44] = 11
    npary[npary == 25] = 12
    npary[npary == 45] = 12
    npary[npary == 26] = 13
    npary[npary == 46] = 13
    npary[npary == 27] = 14
    npary[npary == 47] = 14
    return npary