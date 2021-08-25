import numpy as np
import pandas
import math
import scipy.stats as stats
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


def convolved_smearing(data, x_var, kernel=None, parameters=None):
    if parameters is None and kernel is None:
        parameters = {'mu': 0., 'sigma': 5., 'lower': -100., 'upper': 100.}
    if parameters and kernel is None:
        mu = parameters['mu']
        sigma = parameters['sigma']
        lower = parameters['lower']
        upper = parameters['upper']

        if x_var + '_smeared' in data.columns:
            m_df = data[x_var + '_smeared']
        else:
            m_df = data[x_var].to_numpy()
        truncated_gauss_first = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
        data[x_var + '_smeared'] = m_df + truncated_gauss_first.rvs(len(m_df))
    elif kernel:
        m_df = data[x_var].to_numpy()
        data[x_var + '_smeared'] = m_df + kernel.sample(len(m_df)).numpy().reshape(-1)
    return data


def calc_migration(data, cut, true_cut=None, nobrem=False, x_var='q2'):
    true_var = x_var + '_TRUE'
    if true_cut is None:
        true_cut = cut
    if nobrem:
        y_var = x_var
        x_var = x_var + '_nobrem'
        y_cut = 19.0
    else:
        y_var = x_var + '_nobrem'
        y_cut = 14.0
    num_true_events_in = len(data.query(f'{true_var} > {str(true_cut)} & {y_var} > {str(y_cut)}'))
    num_reco_events_in = len(data.query(f'{x_var} > {str(cut)}'))

    num_events_down = len(
        data.query(f'{true_var} > {str(true_cut)} & {x_var} <= {str(cut)} & {y_var} > {str(y_cut)}').index)
    num_events_up = len(
        data.query(f'{true_var} <= {str(true_cut)} & {x_var} > {str(cut)} & {y_var} > {str(y_cut)}').index)
    num_events_inside = len(
        data.query(f'{true_var} > {str(true_cut)} & {x_var} > {str(cut)} & {y_var} > {str(y_cut)}').index)

    migration_down = num_events_down * 100 / num_true_events_in
    migration_up = num_events_up * 100 / num_true_events_in
    migration_inside = num_events_inside * 100 / num_true_events_in

    print(f'DOWN: {migration_down:.2f} %')
    print(f'UP: {migration_up:.2f} %')
    print(f'STAYED INSIDE: {migration_inside:.2f} %')
