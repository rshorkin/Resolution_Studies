import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
from zfit import z
import zfit.z.numpy as znp
from matplotlib.ticker import AutoMinorLocator, LogLocator, LogFormatterSciNotation
import csv
from scipy.stats import chisquare
import mplhep as hep
import math
import zfit.models.convolution as zconv
import os

from Hist_Settings import hist_dict


# Here are the functions for fitting J/Psi MC, using the fit's shape to fit data (letting some parameters loose)
# and finally getting parameters for smearing from all of this
class CustomTicker(LogFormatterSciNotation):
    def __call__(self, x, pos=None):
        if x not in [1, 10]:
            return LogFormatterSciNotation.__call__(self, x, pos=None)
        else:
            return "{x:g}".format(x=x)


class LogNormal(zfit.pdf.BasePDF):
    def __init__(self, mu, sigma, theta, obs, name='LogNormal', ):
        params = {'mu': mu, 'sigma': sigma, 'theta': theta}
        super().__init__(obs, params, name=name)

    def _unnormalized_pdf(self, x):
        data = z.unstack_x(x)
        mu = self.params['mu']
        sigma = self.params['sigma']
        theta = self.params['theta']

        cond = tf.less_equal(data, theta)
        exp_power = -(znp.power(znp.log(data - theta) - mu, 2) / (2 * znp.power(sigma, 2)))
        outer_factor = (data - theta) * sigma * znp.sqrt(2 * math.pi)
        func = tf.where(cond,
                        0,
                        np.exp(exp_power) / outer_factor)
        return func


# formatting data into zfit-compatible format
def format_data(data, obs):
    return zfit.Data.from_numpy(obs, data.to_numpy())


# service function for parameter naming
def name_tags(tags):
    return f'_{tags["brem_cat"]}_{tags["trig_cat"]}_{tags["run_num"]}_{tags["smeared"]}'


# Creating initial model to fit J/Psi
def create_initial_model(initial_parameters, obs, tags, type='brem_PX'):
    # Double Crystal Ball for nobrem_P
    if 'TRACK' not in type:
        mu = zfit.Parameter(f"mu_{type}" + name_tags(tags), initial_parameters['mu'])
        sigma = zfit.Parameter(f'sigma_{type}' + name_tags(tags), initial_parameters['sigma'])
        alphal = zfit.Parameter(f'alphal_{type}' + name_tags(tags), initial_parameters['alphal'])
        nl = zfit.Parameter(f'nl_{type}' + name_tags(tags), initial_parameters['nl'], 1., 10.)
        ar = zfit.Parameter(f'alphar_{type}' + name_tags(tags), initial_parameters['alphar'])
        nr = zfit.Parameter(f'nr_{type}' + name_tags(tags), initial_parameters['nr'], 1., 10.)

        model = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphar=ar, nr=nr, alphal=alphal, nl=nl)
    elif 'TRACK' in type:
        if 'X' in type or 'Y' in type:
            lmu = zfit.Parameter(f"mu_left_{type}" + name_tags(tags), -initial_parameters['mu'])
            lsigma = zfit.Parameter(f'sigma_left_{type}' + name_tags(tags), initial_parameters['sigma'])
            lal = zfit.Parameter(f'alphal_left_{type}' + name_tags(tags), initial_parameters['alphal'])
            lnl = zfit.Parameter(f'nl_left_{type}' + name_tags(tags), initial_parameters['nl'], 1., 20.)
            lar = zfit.Parameter(f'alphar_left_{type}' + name_tags(tags), initial_parameters['alphar'])
            lnr = zfit.Parameter(f'nr_left_{type}' + name_tags(tags), initial_parameters['nr'], 1., 10.)
            frac = zfit.Parameter(f'frac_{type}' + name_tags(tags), 0.5, 0.01, 0.999)

            LeftDCB = zfit.pdf.DoubleCB(obs=obs, mu=lmu, sigma=lsigma, alphar=lar, nr=lnr, alphal=lal, nl=lnl)

            rmu = zfit.Parameter(f"mu_right_{type}" + name_tags(tags), initial_parameters['mu'])
            rsigma = zfit.Parameter(f'sigma_right_{type}' + name_tags(tags), initial_parameters['sigma'])
            ral = zfit.Parameter(f'alphal_right_{type}' + name_tags(tags), initial_parameters['alphal'])
            rnl = zfit.Parameter(f'nl_right_{type}' + name_tags(tags), initial_parameters['nl'], 1., 10.)
            rar = zfit.Parameter(f'alphar_right_{type}' + name_tags(tags), initial_parameters['alphar'])
            rnr = zfit.Parameter(f'nr_right_{type}' + name_tags(tags), initial_parameters['nr'], 1., 20.)

            RightDCB = zfit.pdf.DoubleCB(obs=obs, mu=rmu, sigma=rsigma, alphar=rar, nr=rnr, alphal=ral, nl=rnl)
            model = zfit.pdf.SumPDF([LeftDCB, RightDCB], fracs=(frac,))

        else:
            mu = zfit.Parameter(f"mu_{type}" + name_tags(tags), initial_parameters['mu'])
            sigma = zfit.Parameter(f'sigma_{type}' + name_tags(tags), initial_parameters['sigma'])
            theta = zfit.Parameter(f'theta_{type}' + name_tags(tags), initial_parameters['theta'])
            model = LogNormal(obs=obs, mu=mu, sigma=sigma, theta=theta)
    else:
        raise Warning('Decide model type!')
    return model


# Minimizing the J/Psi model

def initial_fitter(data, model, obs):
    data = format_data(data, obs)
    # Create NLL
    nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    # Create minimizer
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)
    result = minimizer.minimize(nll)
    if result.valid:
        print("Result is valid")
        print("Converged:", result.converged)
        param_errors = result.hesse()
        params = result.params
        print(params)
        if not result.valid:
            print("Error calculation failed \nResult is not valid")
            return None
        else:
            return {param[0].name: param[1]['value'] for param in result.params.items()}
    else:
        print('Minimization failed \nResult: \n{0}'.format(result))
        return None


# Plotting

def plot_fit_result(models, data, obs, tags, plt_name, pulls_switch=False):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]
    s_tag = tags["smeared"]

    lower, upper = obs.limits

    plot_label = hist_dict[plt_name]["plot_label"] + "\n" + b_tag + "\n" + t_tag
    h_bin_width = hist_dict[plt_name]["bin_width"]
    h_num_bins = hist_dict[plt_name]["num_bins"]
    h_xmin = hist_dict[plt_name]["x_min"]
    h_xmax = hist_dict[plt_name]["x_max"]
    h_xlabel = hist_dict[plt_name]["x_label"]
    h_ylabel = hist_dict[plt_name]["y_label"]
    if 'mee' in plt_name and 'nobrem' not in plt_name:
        title = f'J/$\psi$ mass fit {b_tag}'
    elif 'mKee' in plt_name and 'nobrem' not in plt_name:
        title = f'$B^+$ mass fit {b_tag}'
    elif 'mee' in plt_name and 'nobrem' in plt_name:
        title = f'J/$\psi$ track mass fit {b_tag}'
    elif 'mee' in plt_name and 'nobrem' in plt_name:
        title = f'$B^+$ track mass fit {b_tag}'
    else:
        title = f'{plt_name}, {b_tag}'

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    main_axes.set_title(title, fontsize=18)

    if data is not None:
        data_x, _ = np.histogram(data.values, bins=bins)
        data_errors = np.sqrt(data_x)
        data_sum = data_x.sum()
        plot_scale = data_sum * obs.area() / h_num_bins
        main_axes.errorbar(bin_centers, data_x, yerr=np.sqrt(data_x), fmt="ok", label='data', markersize='4')

    main_axes.set_xlim(h_xmin, h_xmax)
    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_ylabel(h_ylabel)
    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    data_bins = np.linspace(lower[-1][0], upper[0][0], num=h_num_bins + 1)
    colors = ["b", "k", "r"]
    j = 0

    for model_name, model in models.items():
        if model.is_extended:
            if data is None:
                sp_scale = model.get_yield()
            else:
                sp_scale = 1.
            main_axes.plot(x_plot,
                           model.ext_pdf(x_plot) * obs.area() / (h_num_bins * sp_scale),
                           colors[j],
                           label=model_name)
            if pulls_switch and (len(models) == 1 or ('data' in model_name and len(models) == 2)
                                 or ('smeared' in model_name)):
                cond = np.not_equal(data_errors, 0.)
                pulls = np.where(cond,
                                 np.divide(
                                     np.subtract(data_x,
                                                 model.ext_pdf(bin_centers) * obs.area() / h_num_bins),
                                     data_errors),
                                 0.)
        else:
            main_axes.plot(x_plot,
                           model.pdf(x_plot) * plot_scale,
                           colors[j],
                           label=model_name)
            if pulls_switch and (len(models) == 1 or ('data' in model_name and len(models) == 2)
                                 or ('smeared' in model_name)):
                cond = np.not_equal(data_errors, 0.)
                pulls = np.where(cond,
                                 np.divide(
                                     np.subtract(data_x,
                                                 model.pdf(bin_centers) * plot_scale),
                                     data_errors),
                                 0.)
        j = j + 1

    main_axes.legend(title=plot_label, loc="upper left")
    main_axes.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # main_axes.set_yscale('log')
    # bottom = max([min(data_x) * 0.7, 10.])
    # main_axes.set_ylim(bottom=bottom)
    # main_axes.yaxis.set_major_formatter(CustomTicker())
    # locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    # main_axes.yaxis.set_minor_locator(locmin)

    main_axes.set_ylim(bottom=0)
    if not pulls_switch:
        main_axes.set_xlabel(h_xlabel)
    if pulls_switch:
        main_axes.set_xticklabels([])
        # pulls subplot
        plt.axes([0.1, 0.1, 0.85, 0.2])
        plt.yscale("linear")
        pulls_axes = plt.gca()
        xs = [h_xmin]
        ys = [pulls[0]]
        for i in range(h_num_bins - 1):
            xs.append(h_xmin + h_bin_width * (1 + i))
            xs.append(h_xmin + h_bin_width * (1 + i))
            ys.append(pulls[i])
            ys.append(pulls[i + 1])
        del i
        xs.append(h_xmax)
        ys.append(pulls[-1])

        pulls_axes.plot(xs, ys, color=colors[j - 1], label='Pulls')

        pulls_axes.set_ylim(-10., 10.)
        pulls_axes.set_yticks([-5, 0, 5])
        pulls_axes.xaxis.set_minor_locator(AutoMinorLocator())
        y_ticks = pulls_axes.yaxis.get_major_ticks()
        pulls_axes.set_xlabel(h_xlabel)
        pulls_axes.set_ylabel('Pull')
        pulls_axes.set_xlim(h_xmin, h_xmax)
        plt.grid("True", axis="y", color="black", linestyle="--")

    if not os.path.exists(f'../Results/Plots/{plt_name}'):
        os.makedirs(f'../Results/Plots/{plt_name}')
    plt.savefig(f'../Results/Plots/{plt_name}/{plt_name}_fit_plot_{b_tag}_{t_tag}_{r_tag}_{s_tag}.jpg')
    print('saved ' + f'../Results/Plots/{plt_name}/{plt_name}_fit_plot_{b_tag}_{t_tag}_{r_tag}_{s_tag}.jpg')
    # plt.show()
    plt.close()


# Fit data with shape from MC. Allow mean shift and width scale factor to float.
# Additional functions for composed parameter initialization


def write_to_csv(parameters, tags, plt_name):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]

    with open(str("../CSV-Output/{0}/{0}_fit_{1}_{2}_run_{3}.csv".format(plt_name, b_tag, t_tag, r_tag)),
              mode="w") as data_file:
        key_names = [key for key in parameters.keys()]
        data_writer = csv.DictWriter(data_file, fieldnames=key_names)
        data_writer.writeheader()
        for index in range(len(parameters[key_names[0]])):
            data_writer.writerow({key: parameters[key] for key in key_names})


def convoluted_data_model(mc_pdf, data, tags, obs, x_var):
    num_events = len(data.index)
    data = format_data(data, obs)
    if 'brem' in x_var:
        lower = -1000
        upper = 1000
        mu = 150
        sigma = 150.
    elif 'TRACK' in x_var:
        lower = -5000
        upper = 5000
        mu = -1400.
        sigma = 100.
    # Create kernel for convolution
    mu_g = zfit.Parameter(f"conv_ker_mu_{x_var}{name_tags(tags)}", mu)
    sigma_g = zfit.Parameter(f'conv_ker_sigma_{x_var}{name_tags(tags)}', sigma)

    obs_kernel = zfit.Space(obs.obs, limits=(lower, upper))

    kernel = zfit.pdf.Gauss(mu=mu_g, sigma=sigma_g, obs=obs_kernel)

    # Convolve pdf used to fit MC with gaussian kernel
    conv_model = zconv.FFTConvPDFV1(mc_pdf, kernel)
    # Try to fit data with it?
    nll = zfit.loss.UnbinnedNLL(model=conv_model, data=data)
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)

    params = [mu_g, sigma_g]

    result = minimizer.minimize(nll, params=params)
    print("Result Valid:", result.valid)
    print("Fit converged:", result.converged)
    print(result.params)
    parameters = {'mu': zfit.run(mu_g), 'sigma': zfit.run(sigma_g), 'lower': lower, 'upper': upper}
    return conv_model, parameters, kernel

