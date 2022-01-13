# /usr/bin/env python

from numba import jit, jit_module, prange
import numpy as np

@jit(nopython=True)
def roll_1D(x, p):
    """
    Function to shift traces stored in a 1D array (x)
    by the number of points stored in 1D array (p).
    optimised in Numba.

    """

    p = p*-1
    x = np.append(x[p:], x[:p])
    return x

@jit(nopython=True)
def roll_2D(array, shifts):
    """
    Function to shift traces stored in a 2D array (x)
    by the number of points stored in 1D array (p).
    optimised in Numba.

    """

    n = array.shape[0]
    array_new = np.copy(array)
    for i in range(n):
        array_new[int(i)] = roll_1D(array_new[int(i)],int(shifts[int(i)]))
        # array_new[i] = np.roll(array[i],int(shifts[i]))


    return array_new

@jit(nopython=True)
def my_sum(array):
    """
    Function to sum a 1D array.
    """

    total = 0

    for i in range(len(array)):
        total += array[i]

    return total



@jit(nopython=True)
def stack_2D(t):
    """
    Takes the mean across a 2D array (t) as if 'stacking'
    """
    nt = t.shape[0]
    lt = t.shape[1]
    sumt = np.zeros(lt)
    for ti in t:
        sumt += ti
    meant = sumt / nt
    return meant
