import numpy as np
import theano.tensor as tt

# Functions for efficient coding model
def f(v, mu, s):
    return 0.5 * (1 + np.tanh((v - mu) / (2 * s)))


def f_theano(v, mu, s):
    return 0.5 * (1 + tt.tanh((v - mu) / (2 * s)))


def f_inv(v_tilde, mu, s):
    # v_tilde = np.clip(v_tilde, 1e-8, 1 - 1e-8)
    v_tilde[(v_tilde < 1e-8) | (v_tilde > 1 - 1e-8)] = np.nan
    result = mu + s * np.log(v_tilde / (1 - v_tilde))
    return result


def phi_prime(v_tilde, s):
    # The first derivative of the inverse CDF
    return s / (v_tilde * (1 - v_tilde))


def phi_double_prime(v_tilde, s):
    # The second derivative of the inverse CDF
    return s * (2 * v_tilde - 1) / (v_tilde ** 2 * (1 - v_tilde) ** 2)


def f_prime_theano(v, mu, s):
    p = f_theano(v, mu, s)
    return p * (1 - p)


def g(v_hat):
    return 1 / (1 + np.exp(-v_hat))


def g_inv_theano(v):
    return tt.log(v / (1 - v))


def g_inv_prime_theano(v):
    return 1 / (v * (1 - v))


def sech(x):
    '''define the sech function using theano'''
    return 2 / (tt.exp(x) + tt.exp(-x))


def logp_logistic(v, mu, s):
    # logistic PDF transformed to log-probability
    #  Define the log probability function for the logistic distribution
    return tt.log(sech((v - mu) / (2 * s)) ** 2 / (4 * s))


def sech_bis(x):
    # Adaptation of functions using theanos, with numpy for visualization of prior distribution
    return 2 / (np.exp(x) + np.exp(-x))


def logp_logistic_bis(v, mu, s):
    return np.log(sech_bis((v - mu) / (2 * s)) ** 2 / (4 * s))


def normal_cdf(x):
    """ Compute the CDF of the standard normal distribution at x. """
    return 0.5 * (1 + tt.erf(x / tt.sqrt(2)))
