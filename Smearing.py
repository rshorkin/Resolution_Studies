import numpy as np
import pandas
import math
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


def convolved_smearing(data, x_var, parameters=None):
    if parameters is None:
        parameters = {'mu': 0., 'sigma': 5.}
    mu = parameters['mu']
    sigma = parameters['sigma']

    m_df = data[x_var].to_numpy()
    data[x_var + '_smeared'] = m_df + np.random.normal(loc=mu, scale=sigma)

    return data
