import numpy as np
import pandas
import math
from scipy import stats, signal
import matplotlib.pyplot as plt
from pandas import Series, DataFrame


# ================================== #
# This file hosts smearing functions #
# ================================== #

def smearing_fn(m, m_true, scale_sigma, delta_mu, mu, particle_mass):
    return m_true + scale_sigma * (m - m_true) + delta_mu + (1 - scale_sigma) * (mu - particle_mass)


def smear(data, parameters, x_var):
    mu = parameters["mu"]
    delta_mu = parameters["delta_mu"]
    scale_sigma = parameters["scale_sigma"]
    mass = parameters["mass"]

    m_df = data[x_var].to_numpy()
    m_true_df = data[x_var + "_TRUE"].to_numpy()

    data[x_var + "_smeared"] = np.vectorize(smearing_fn)(m_df, m_true_df, scale_sigma, delta_mu, mu, mass)
    return data


def convolve_smear(data, x_var, parameters=None):
    if parameters is None:
        parameters = {'mu': 0., 'sigma': 1.}
    mu = parameters['mu']
    sigma = parameters['sigma']

    m_df = data[x_var].to_numpy()
    num_points = len(m_df)

    gaussian = np.random.normal(loc=mu, scale=sigma, size=num_points)
    bins = np.linspace(300, 3300, num=num_points)
    print(gaussian)

    gauss_h, _ = np.histogram(gaussian, bins=num_points)
    mc_hist, _ = np.histogram(m_df, bins=num_points)

    smeared_result = signal.fftconvolve(mc_hist, gauss_h, 'same')
    smeared_result_n = smeared_result*num_points

    # data_smeared = np.convolve(gaussian, m_df, 'same')
    # data_smeared = scipy.signal.convolve(m_df, gaussian)
    # print(len(smeared_result))
    # plt.hist(smeared_result, bins=50, label='convoluted stuff')
    # plt.hist(gaussian_normed, bins=50, label='mc')
    plt.show()
    plt.clf()
    smth = plt.plot(bins, smeared_result)
    plt.show()
    plt.clf()
    plt.hist(m_df, bins=50, label='convoluted stuff')
    plt.show()
    plt.clf()
    plt.hist(gaussian, bins=50, label='convoluted stuff')
    plt.show()
    # plt.hist(smeared_result_n, label='convoluted stuff')
    # plt.hist(gaussian_normed, bins=50, label='mc')
    # return smeared_result


