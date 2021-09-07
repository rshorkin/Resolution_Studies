import numpy as np
import csv
import pandas
import math
import scipy.stats as stats
from pandas import Series, DataFrame

import Service
from Fitting import name_tags
import zfit


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


def calc_migration(data, cut, true_cut=None, nobrem=False):
    if nobrem:
        x_var = 'J_psi_1S_TRACK_M'
        sm_var = x_var + '_smeared'
        data["q2_nobrem_smeared"] = data[sm_var].apply(Service.calc_q2)
        data["q2_nobrem_smeared"] = data["q2_nobrem_smeared"].apply(Service.to_GeVsq)
        x_var = 'q2_nobrem'
    else:
        x_var = 'J_psi_1S_M'
        sm_var = x_var + '_smeared'
        data["q2_smeared"] = data[sm_var].apply(Service.calc_q2)
        data["q2_smeared"] = data["q2_smeared"].apply(Service.to_GeVsq)
        x_var = 'q2'
    if true_cut is None:
        true_cut = cut
    true_var = 'q2_TRUE'
    sm_var = x_var + '_smeared'
    num_true_events_in = len(data.query(f'{true_var} > {str(true_cut)}'))

    num_events_down = len(
        data.query(f'{true_var} > {str(true_cut)} & {x_var} <= {str(cut)}').index)
    num_events_up = len(
        data.query(f'{true_var} <= {str(true_cut)} & {x_var} > {str(cut)} ').index)
    num_events_inside = len(
        data.query(f'{true_var} > {str(true_cut)} & {x_var} > {str(cut)}').index)

    num_sm_events_down = len(
        data.query(f'{true_var} > {str(true_cut)} & {sm_var} <= {str(cut)}').index)
    num_sm_events_up = len(
        data.query(f'{true_var} <= {str(true_cut)} & {sm_var} > {str(cut)} ').index)
    num_sm_events_inside = len(
        data.query(f'{true_var} > {str(true_cut)} & {sm_var} > {str(cut)}').index)

    migration_down = num_events_down * 100 / num_true_events_in
    migration_up = num_events_up * 100 / num_true_events_in
    migration_inside = num_events_inside * 100 / num_true_events_in

    print('------')
    print(f'DOWN: {migration_down:.2f} %')
    print(f'UP: {migration_up:.2f} %')
    print(f'STAYED INSIDE: {migration_inside:.2f} %')

    sm_migration_down = num_sm_events_down * 100 / num_true_events_in
    sm_migration_up = num_sm_events_up * 100 / num_true_events_in
    sm_migration_inside = num_sm_events_inside * 100 / num_true_events_in

    print(f'SMEARING DOWN: {sm_migration_down:.2f} %')
    print(f'SMEARING UP: {sm_migration_up:.2f} %')
    print(f'SMEARING STAYED INSIDE: {sm_migration_inside:.2f} %')
    print('------')

    return {'in -> down, [%]': f"{migration_down:.2f}", 'in -> in, [%]': f"{migration_inside:.2f}",
            'down -> in, [%]': f"{migration_up:.2f}"}, \
           {'in -> down, [%]': f"{sm_migration_down:.2f}", 'in -> in, [%]': f"{sm_migration_inside:.2f}",
            'up -> in, [%]': f"{sm_migration_up:.2f}"}


def smear_from_csv(data, x_var, tags, naming, add=''):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]
    path = f'../Results/Parameters/{naming}/{b_tag}_{t_tag}_{r_tag}.csv'
    params = {}
    with open(path, 'r') as file:
        for line in csv.reader(file):
            params[line[0]] = float(line[1])
    lower = params['lower']
    upper = params['upper']
    obs_kernel = zfit.Space('kernel obs', limits=(lower, upper))
    if 'mu_g' and 'mu_dcb' in params.keys():
        mu_g = zfit.Parameter(f"conv_ker_mu{name_tags(tags)}{add}", params['mu_g'])
        sigma_g = zfit.Parameter(f'conv_ker_sigma{name_tags(tags)}{add}', params['sigma_g'])
        al = zfit.Parameter(f'conv_ker_al{name_tags(tags)}{add}', params['al'])
        ar = zfit.Parameter(f'conv_ker_ar{name_tags(tags)}{add}', params['ar'])
        nl = zfit.Parameter(f'conv_ker_nl{name_tags(tags)}{add}', params['nl'])
        nr = zfit.Parameter(f'conv_ker_nr{name_tags(tags)}{add}', params['nr'])

        mu_ad = zfit.Parameter(f"conv_ker_mu_ad{name_tags(tags)}{add}", params['mu_dcb'])
        sigma_ad = zfit.Parameter(f'conv_ker_sigma_ad{name_tags(tags)}{add}', params['sigma_dcb'])
        frac = zfit.Parameter('kernel_frac' + name_tags(tags) + f'_{add}', params['frac'])

        if tags["brem_cat"] != 'brem_zero':
            DCB = zfit.pdf.DoubleCB(mu=mu_ad, sigma=sigma_ad, alphal=al, alphar=ar, nl=nl, nr=nr, obs=obs_kernel)
            gauss = zfit.pdf.Gauss(mu=mu_g, sigma=sigma_g, obs=obs_kernel)
            kernel = zfit.pdf.SumPDF([DCB, gauss], fracs=[frac])
        else:
            kernel = zfit.pdf.DoubleCB(mu=mu_ad, sigma=sigma_ad, alphal=al, alphar=ar, nl=nl, nr=nr, obs=obs_kernel)

    elif 'mu' in params.keys():
        mu = zfit.Parameter(f"conv_ker_mu{name_tags(tags)}{add}", params['mu'])
        sigma = zfit.Parameter(f'conv_ker_sigma{name_tags(tags)}{add}', params['sigma'])

        kernel = zfit.pdf.Gauss(mu=mu, sigma=sigma, obs=obs_kernel)

    m_df = data[x_var].to_numpy()
    data[x_var + '_smeared'] = m_df + kernel.sample(len(m_df)).numpy().reshape(-1)
    return data


