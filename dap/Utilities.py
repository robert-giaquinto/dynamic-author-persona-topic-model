from __future__ import division, print_function, absolute_import
from math import log, exp
import numpy as np
import cPickle as pickle
from scipy.special import psi


def digamma(x):
    x = x + 6
    p = 1 / (x * x)
    p = (((0.004166666666667 * p - 0.003968253986254) * p + 0.008333333333333) * p - 0.083333333333333) * p
    p = p + safe_log(x) - 0.5 / x - 1 / (x - 1) - 1 / (x - 2) - 1 / (x - 3) - 1 / (x - 4) - 1 / (x - 5) - 1 / (x - 6)
    return p


def dirichlet_expectation(dirichlet_parameter):
    """
    compute dirichlet expectation
    :param dirichlet_parameter:
    :return:
    """
    if len(dirichlet_parameter.shape) == 1:
        return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter))
    return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter, 1))[:, np.newaxis]


def log_sum(log_a, log_b):
    if log_a == -1:
        return log_b

    if log_a < log_b:
        v = log_b + log(1 + exp(log_a-log_b))
    else:
        v = log_a + log(1 + exp(log_b-log_a))

    return v


def safe_log(x):
    if x == 0:
        rval = -100
    else:
        rval = log(x)
    return rval


def softmax(x, axis):
    """
    Softmax for normalizing a matrix along an axis
    Use max substraction approach for numerical stability
    :param x:
    :return:
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def safe_log_array(arr):
    rval = np.log(np.clip(arr, a_min=1e-10, a_max=0.99999))
    return rval


def pickle_it(obj, filepath):
    outfile = open(filepath, "wb")
    pickle.dump(obj, outfile)
    outfile.close()


def unpickle_it(filepath):
    infile = open(filepath, "rb")
    obj = pickle.load(infile)
    infile.close()
    return obj



def matrix2str(mat, num_digits=2):
    """
    take a matrix (either list of lists of numpy array) and put it in a
    pretty printable format.
    :param mat: matrix to print
    :param num_digits: how many significant digits to show
    :return:
    """
    rval = ''
    for row in mat:
        s = '{:.' + str(num_digits) + '}'
        # rval += '\t'.join([s.format(round(elt, num_digits)) for elt in row]) + '\n'
        fpad = ['' if round(elt, num_digits) < 0 else ' ' for elt in row]
        bpad = [' ' * (1 + 7 - len(str(np.abs(round(elt, num_digits))))) for elt in row]
        rval += ''.join([f + s.format(round(elt, num_digits)) + b for elt, f, b in zip(row, fpad, bpad)]) + '\n'
    return rval
