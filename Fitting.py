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

        cond = tf.less_equal(data, -theta)
        exp_power = -(znp.power(znp.log(-data - theta) - mu, 2) / (2 * znp.power(sigma, 2)))
        outer_factor = (-data - theta) * sigma * znp.sqrt(2 * math.pi)
        func = tf.where(cond,
                        np.exp(exp_power) / outer_factor,
                        0.)
        return func


# formatting data into zfit-compatible format
def format_data(data, obs):
    return zfit.Data.from_numpy(obs, data.to_numpy())


# service function for parameter naming
def name_tags(tags):
    return "_{0}_{1}_{2}".format(tags["brem_cat"], tags["trig_cat"], tags["run_num"])


def y(data, mu, theta, sigma):
    exp_power = -(np.power(np.log(-data - theta) - mu, 2) / (2 * np.power(sigma, 2)))
    outer_factor = (-data - theta) * sigma * np.sqrt(2 * math.pi)
    return np.exp(exp_power) / outer_factor


def test_model():
    mu = 7.
    theta = -3200.
    sigma = .5
    x = np.linspace(300., 3200., num=1000)
    plt.plot(x, y(x, mu, theta, sigma))
    plt.show()


# Creating initial model to fit J/Psi
def create_initial_model(initial_parameters, obs, tags, switch="brem", params=None):
    # Double Crystal Ball for q2
    if switch == "brem":

        mu = zfit.Parameter("mu" + name_tags(tags), initial_parameters['mu'],
                            initial_parameters['mu'] - 200., initial_parameters['mu'] + 200.)
        sigma = zfit.Parameter('sigma' + name_tags(tags), initial_parameters['sigma'], 1., 100.)
        alphal = zfit.Parameter('alphal' + name_tags(tags), initial_parameters['alphal'], 0.01, 5.)
        nl = zfit.Parameter('nl' + name_tags(tags), initial_parameters['nl'], 0.1, 400.)
        alphar = zfit.Parameter('alphar' + name_tags(tags), initial_parameters['alphar'], 0.01, 10.)
        nr = zfit.Parameter('nr' + name_tags(tags), initial_parameters['nr'], 0.1, 100.)

        model = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)

    # gauss + exp for q2 no brem
    elif switch == "nobrem":
        mu_l = zfit.Parameter("l_mu_CB" + name_tags(tags), 2292., 500., 3200.)
        sigma_l = zfit.Parameter('l_sigma_CB' + name_tags(tags), 280., 0., 700.)
        alpha_l = zfit.Parameter('l_alpha_CB' + name_tags(tags), 1.7, 0.0001, 10, )
        n_l = zfit.Parameter('l_n_CB' + name_tags(tags), 73.)

        mu_g = zfit.Parameter("g_mu" + name_tags(tags), 2000., 1200., 3100.)
        sigma_g = zfit.Parameter('g_sigma' + name_tags(tags), 150., 0., 700.)

        mu_r = zfit.Parameter("r_mu_DCB" + name_tags(tags), 1504., 1200., 3100.)
        sigma_r = zfit.Parameter('r_sigma_DCB' + name_tags(tags), 367., 2., 600.)
        ar_r = zfit.Parameter('alphar_DCB' + name_tags(tags), 3., 0.01, 10.)
        nr_r = zfit.Parameter('nr_DCB' + name_tags(tags), 18., 0.01, 35.)
        al_r = zfit.Parameter('alphal_DCB' + name_tags(tags), 0.05, 0.01, 10.)
        nl_r = zfit.Parameter('nl_DCB' + name_tags(tags), 10., 0.01, 35.)

        Left_CB = zfit.pdf.CrystalBall(obs=obs, mu=mu_l, sigma=sigma_l, alpha=alpha_l, n=n_l)
        Gauss = zfit.pdf.Gauss(obs=obs, mu=mu_g, sigma=sigma_g)
        DCB = zfit.pdf.DoubleCB(obs=obs, mu=mu_r, sigma=sigma_r, alphal=al_r, nl=nl_r, alphar=ar_r, nr=nr_r)
        Right_CB = zfit.pdf.CrystalBall(obs=obs, mu=mu_r, sigma=sigma_r, alpha=ar_r, n=nr_r)

        mu = zfit.Parameter('mu', 6.)
        theta = zfit.Parameter('theta', -3160.)
        sigma = zfit.Parameter('sigma', .6)
        Log_n = LogNormal(obs=obs, mu=mu, sigma=sigma, theta=theta)

        frac1 = zfit.Parameter('frac1' + name_tags(tags), 0.75, 0.01, .99)
        frac2 = zfit.Parameter('frac2' + name_tags(tags), 0.15, 0.01, .99)

        model = zfit.pdf.SumPDF([Log_n, Left_CB, Right_CB], fracs=[frac1, frac2])

    elif switch == "smearedMC":
        mu = zfit.Parameter("mu" + name_tags(tags), initial_parameters['mu'],
                            initial_parameters['mu'] - 200., initial_parameters['mu'] + 200.)
        sigma = zfit.Parameter('sigma' + name_tags(tags), initial_parameters['sigma'], 1., 100.)
        alphal = zfit.Parameter('alphal' + name_tags(tags), initial_parameters['alphal'], 0.01, 5.)
        nl = zfit.Parameter('nl' + name_tags(tags), initial_parameters['nl'], 0.1, 400.)

        alphar = zfit.Parameter('alphar' + name_tags(tags),
                                params['alphar']['value'],
                                floating=False)
        nr = zfit.Parameter('nr' + name_tags(tags),
                            params['nr']['value'],
                            floating=False)

        model = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)

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

def plot_fit_result(models, data, obs, tags, plt_name, smeared=None, pulls_switch=False):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]
    if smeared:
        tags["run_num"] = "run2_smeared"

    lower, upper = obs.limits

    plot_label = hist_dict[plt_name]["plot_label"] + "\n" + b_tag + "\n" + t_tag
    h_bin_width = hist_dict[plt_name]["bin_width"]
    h_num_bins = hist_dict[plt_name]["num_bins"]
    h_xmin = hist_dict[plt_name]["x_min"]
    h_xmax = hist_dict[plt_name]["x_max"]
    h_xlabel = hist_dict[plt_name]["x_label"]
    h_ylabel = hist_dict[plt_name]["y_label"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    main_axes.set_title('M(ee) without brem recovery, J/$\psi$ MC, brem one', fontsize=18)

    if data is not None:
        data_x, _ = np.histogram(data.values, bins=bins)
        data_errors = np.sqrt(data_x)
        data_sum = data_x.sum()
        plot_scale = data_sum * obs.area() / h_num_bins
        main_axes.errorbar(bin_centers, data_x, yerr=np.sqrt(data_x), fmt="ok", label=tags["sample"], markersize='4')

    main_axes.set_xlim(h_xmin, h_xmax)
    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_ylabel(h_ylabel)
    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    data_bins = np.linspace(lower[-1][0], upper[0][0], num=h_num_bins + 1)
    colors = ["b", "k", "r"]
    i = 0
    for model_name, model in models.items():
        if model.is_extended:
            main_axes.plot(x_plot,
                           model.ext_pdf(x_plot) * obs.area() / (h_num_bins * model.get_yield()),
                           colors[i],
                           label=model_name)
            if pulls_switch:
                pulls = np.divide(
                    np.subtract(data_x,
                                model.ext_pdf(bin_centers) * obs.area() / (h_num_bins * model.get_yield())),
                    data_errors)
        else:
            main_axes.plot(x_plot,
                           model.pdf(x_plot) * plot_scale,
                           colors[i],
                           label=model_name)
            if pulls_switch:
                pulls = np.divide(
                    np.subtract(data_x,
                                model.pdf(bin_centers) * plot_scale),
                    data_errors)
        i = i + 1
    main_axes.legend(title=plot_label, loc="upper left")
    main_axes.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    # main_axes.set_yscale('log')
    # bottom = max([min(data_x) * 0.7, 10.])
    # main_axes.set_ylim(bottom=bottom)
    # main_axes.yaxis.set_major_formatter(CustomTicker())
    # locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
    # main_axes.yaxis.set_minor_locator(locmin)

    main_axes.set_ylim(bottom=0)
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
        xs.append(h_xmax)
        ys.append(pulls[-1])

        pulls_axes.plot(xs, ys, color='blue', label='Pulls')

        pulls_axes.set_ylim(-10., 10.)
        pulls_axes.set_yticks([-5, 0, 5])
        pulls_axes.xaxis.set_minor_locator(AutoMinorLocator())
        y_ticks = pulls_axes.yaxis.get_major_ticks()
        pulls_axes.set_xlabel(h_xlabel)
        pulls_axes.set_ylabel('Pull')
        pulls_axes.set_xlim(h_xmin, h_xmax)
        plt.grid("True", axis="y", color="black", linestyle="--")
    plt.savefig(f"../Output/{plt_name}/{plt_name}_fit_plot_{b_tag}_{t_tag}_{r_tag}.jpg")
    # plt.show()
    plt.close()


# Fit data with shape from MC. Allow mean shift and width scale factor to float.
# Additional functions for composed parameter initialization

def mu_shifted_fn(x, dx):
    return x + dx


def sigma_scaled_fn(x, scale_x):
    return x * scale_x


def n_scaled_fn(x, scale_x):
    return x / scale_x


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


# data fitting and getting smearing parameters
def create_data_fit_model(data, parameters, obs, tags):
    num_events = len(data.index)
    data = format_data(data, obs)
    b_tag = tags["brem_cat"]
    # Initializing new model parameters to save the previous model for comparison
    # Floating parameters, required for smearing
    shift_mu = zfit.Parameter('delta_mu' + name_tags(tags), 0., -200., 200.)
    scale_sigma = zfit.Parameter('scale_sigma' + name_tags(tags), 1., 0.001, 100.)

    # main fit parameters, not allowed to float (though we explicitly say which parameters to float later)
    mu = zfit.Parameter('data_mu' + name_tags(tags),
                        parameters['mu' + name_tags(tags)],
                        floating=False)
    sigma = zfit.Parameter('data_sigma' + name_tags(tags),
                           parameters['sigma' + name_tags(tags)],
                           floating=False)
    alphal = zfit.Parameter('data_alphal' + name_tags(tags),
                            parameters['alphal' + name_tags(tags)],
                            floating=False)
    nl = zfit.Parameter('data_nl' + name_tags(tags),
                        parameters['nl' + name_tags(tags)],
                        floating=False)
    alphar = zfit.Parameter('data_alphar' + name_tags(tags),
                            parameters['alphar' + name_tags(tags)],
                            floating=False)
    nr = zfit.Parameter('data_nr' + name_tags(tags),
                        parameters['nr' + name_tags(tags)],
                        floating=False)
    # Additional floating parameter, required for better simulation of an upper power law tail, only for brem 1, 2
    if b_tag == "b_zero":
        scale_r = zfit.Parameter('sc_r' + name_tags(tags), 1., floating=False)
    else:
        scale_r = zfit.Parameter('sc_r' + name_tags(tags), 1., 0.01, 2.)

    # Create composed parameters
    mu_shifted = zfit.ComposedParameter("mu_shifted" + name_tags(tags),
                                        mu_shifted_fn,
                                        params=[mu, shift_mu])
    sigma_scaled = zfit.ComposedParameter("sigma_scaled" + name_tags(tags),
                                          sigma_scaled_fn,
                                          params=[sigma, scale_sigma])
    nr_scaled = zfit.ComposedParameter("nr_scaled" + name_tags(tags),
                                       n_scaled_fn,
                                       params=[nr, scale_r])
    alphar_scaled = zfit.ComposedParameter("alphar_scaled" + name_tags(tags),
                                           sigma_scaled_fn,  # used as a general multiplication function
                                           params=[alphar, scale_r])

    # Creating model with new scale/shift parameters
    model = zfit.pdf.DoubleCB(obs=obs, mu=mu_shifted, sigma=sigma_scaled,
                              alphal=alphal, nl=nl, alphar=alphar_scaled, nr=nr_scaled)

    # Background model: exponential
    lambd = zfit.Parameter("lambda" + name_tags(tags), -0.00005, -1., 0.)
    model_bgr = zfit.pdf.Exponential(lambd, obs=obs)

    # Make models extended and combine them
    n_sig = zfit.Parameter("n_signal" + name_tags(tags),
                           int(num_events * 0.99), int(num_events * 0.6), int(num_events * 1.2), step_size=1)
    n_bgr = zfit.Parameter("n_bgr" + name_tags(tags),
                           int(num_events * 0.01), 0., int(num_events * 0.4), step_size=1)

    model_extended = model.create_extended(n_sig)
    model_bgr_extended = model_bgr.create_extended(n_bgr)

    model = zfit.pdf.SumPDF([model_extended, model_bgr_extended])
    # NLL and minimizer
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)

    # minimization of shift and scale factors
    if b_tag == "brem_zero" or b_tag == 'brem_one' or b_tag == 'brem_two':
        result = minimizer.minimize(nll, params=[lambd, n_sig, n_bgr, mu_shifted, sigma_scaled])
    else:
        result = minimizer.minimize(nll, params=[lambd, n_sig, n_bgr, mu_shifted, sigma_scaled, scale_r])
    final_params = result.params
    param_errors = result.hesse()
    print("Result Valid:", result.valid)
    print("Fit converged:", result.converged)
    print(result.params)
    models = {"combined": model, "signal": model_extended, "background": model_bgr_extended}  # need for tests
    parameters = {param[0].name: {"value": param[1]['value'], "error": err[1]['error']}
                  for param, err in zip(result.params.items(), param_errors.items())}
    parameters["nr"] = {'value': zfit.run(nr_scaled)}
    parameters['alphar'] = {'value': zfit.run(alphar_scaled)}
    return parameters, models


def convoluted_data_model(mc_pdf, data, obs):
    num_events = len(data.index)
    data = format_data(data, obs)
    # Create kernel for convolution
    mu_g = zfit.Parameter("cg_mu", -47.)
    sigma_g = zfit.Parameter('cg_sigma', 61.)
    obs_kernel = zfit.Space('mee_nobrem', limits=(-150., 150.))
    gauss_kernel = zfit.pdf.Gauss(mu=mu_g, sigma=sigma_g, obs=obs_kernel)
    # Convolve pdf used to fit MC with gaussian kernel
    conv_model = zconv.FFTConvPDFV1(mc_pdf, gauss_kernel, limits_func=(300, 3300))
    # Try to fit data with it?
    n_sig = zfit.Parameter("n_events",
                           int(num_events * 0.99), int(num_events * 0.6), int(num_events * 1.2), step_size=1)

    nll = zfit.loss.UnbinnedNLL(model=conv_model, data=data)
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)
    result = minimizer.minimize(nll, params=[mu_g, sigma_g])
    print("Result Valid:", result.valid)
    print("Fit converged:", result.converged)
    print(result.params)

    parameters = {'mu': zfit.run(mu_g), 'sigma': zfit.run(sigma_g)}
    return conv_model, parameters


def combine_models(models, data, obs, tags):
    num_events = len(data.index)
    data = format_data(data, obs)
    tags["brem_cat"] = ""

    if not models["brem_zero"].is_extended:

        n_zero = zfit.Parameter("n_zero" + tags["sample"] + name_tags(tags),
                                int(num_events * 0.25), int(num_events * 0.1), int(num_events * 0.4), step_size=1)
        n_one = zfit.Parameter("n_one" + tags["sample"] + name_tags(tags),
                               int(num_events * 0.5), int(num_events * 0.3), int(num_events * 0.7), step_size=1)
        n_two = zfit.Parameter("n_two" + tags["sample"] + name_tags(tags),
                               int(num_events * 0.25), int(num_events * 0.1), int(num_events * 0.4), step_size=1)

        models["brem_zero"] = models["brem_zero"].create_extended(n_zero)
        models["brem_one"] = models["brem_one"].create_extended(n_one)
        models["brem_two"] = models["brem_two"].create_extended(n_two)

        model = zfit.pdf.SumPDF([models["brem_zero"], models["brem_one"], models["brem_two"]])

        nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
        minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)

        result = minimizer.minimize(nll, params=[n_zero, n_one, n_two])
    else:
        frac_0 = zfit.Parameter("frac_zero" + tags["sample"] + name_tags(tags), 0.25, 0.1, 0.8)
        frac_1 = zfit.Parameter("frac_one" + tags["sample"] + name_tags(tags), 0.5, 0.1, 0.8)
        model = zfit.pdf.SumPDF([models["brem_zero"], models["brem_one"], models["brem_two"]], fracs=[frac_0, frac_1])
        n_yeild = zfit.Parameter("yield" + tags["sample"] + name_tags(tags),
                                 num_events, 0., num_events * 1.2, step_size=1)
        model = model.create_extended(n_yeild)
        nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
        minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)

        result = minimizer.minimize(nll, params=[frac_0, frac_1])

    param_errors = result.hesse()
    print("Result Valid:", result.valid)
    print("Fit converged:", result.converged)
    print(result.params)

    return model
