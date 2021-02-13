import tensorflow as tf
import zfit
import ROOT
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from Hist_Settings import hist_dict

# Here are the functions for fitting J/Psi MC, using the fit's shape to fit data (letting some parameters loose)
# and finally getting parameters for smearing from all of this

obs = zfit.Space('J_psi_1S_M', limits=(2200, 3800))


# formatting data into zfit-compatible format

def format_data(data):
    return zfit.Data.from_numpy(obs, data["J_psi_1S_M"].to_numpy())


# service function for parameter naming
def name_tags(tag_list):
    return "_{0}_{1}_{2}".format(tag_list[0], tag_list[1], tag_list[2])


# Creating initial model to fit J/Psi
def create_initial_model(tag_list):
    initial_params = {'mu': 3097., 'sigma': 20., 'alphal': 0.15, 'nl': 50., 'alphar': 0.9, 'nr': 3.}

    # Double Crystal Ball for each category, like in RK

    mu = zfit.Parameter("mu" + name_tags(tag_list), initial_params['mu'], 3000., 3200.)
    sigma = zfit.Parameter('sigma' + name_tags(tag_list), initial_params['sigma'], 1., 100.)
    alphal = zfit.Parameter('alphal' + name_tags(tag_list), initial_params['alphal'], 0., 5.)
    nl = zfit.Parameter('nl' + name_tags(tag_list), initial_params['nl'], 0., 200.)
    alphar = zfit.Parameter('alphar' + name_tags(tag_list), initial_params['alphar'], 0., 5.)
    nr = zfit.Parameter('nr' + name_tags(tag_list), initial_params['nr'], 0., 200.)

    model = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)
    return model


# Minimizing the J/Psi model

def initial_fitter(data, model):
    data = format_data(data)
    # Create NLL
    nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    # Create minimizer
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)
    result = minimizer.minimize(nll)
    if result.valid:
        print(f'>>> Result is valid')
        print("Converged:", result.converged)
        param_hesse = result.hesse()  # calculate errors todo: nans
        params = result.params
        print(params)
        if not result.valid:
            print(f'>>> Error calculation failed \n>>> Result is not valid')
            return None
        else:
            return {param[0].name: param[1]['value'] for param in result.params.items()}
    else:
        print(f'>>> Minimization failed \n>>> Result: \n{result}')
        return None


# Plotting

def plot_mc_fit_result(model, data, tag_list):
    b_tag = tag_list[0]
    t_tag = tag_list[1]
    r_tag = tag_list[2]

    lower, upper = obs.limits

    plot_label = "$B \\rightarrow KJ/\\psi$\n" + b_tag + "\n" + t_tag

    h_bin_width = hist_dict["bin_width"]
    h_num_bins = hist_dict["num_bins"]
    h_xmin = hist_dict["x_min"]
    h_xmax = hist_dict["x_max"]
    h_xlabel = hist_dict["x_label"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    data_x, _ = np.histogram(data.values, bins=bins)
    data_sum = data_x.sum()
    plot_scale = data_sum * obs.area() / 50

    plt.clf()
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.errorbar(bin_centers, data_x, xerr=16, fmt="ok", label="MC")

    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel("Events/32 MeV")

    main_axes.set_xlabel(h_xlabel)
    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    main_axes.plot(x_plot, model.pdf(x_plot) * plot_scale, color='xkcd:blue', label="mc fit")
    main_axes.legend(title=plot_label, loc="best")
    plt.savefig("../Output/mc_fit_plot_{0}_{1}_run_{2}.pdf".format(b_tag, t_tag, r_tag))
    plt.close()


# Fit data with shape from MC. Allow mean shift and width scale factor to float.
# Additional functions for composed parameter initialization

def mu_shifted_fn(x, dx):
    return x + dx


def sigma_scaled_fn(x, scale_x):
    return x * scale_x


def n_scaled_fn(x, scale_x):
    return x / scale_x


# data fitting and getting smearing parameters
def create_data_fit_model(data, parameters, tag_list):
    b_tag = tag_list[0]
    data = format_data(data)
    # Initializing new model parameters to save the previous model for comparison
    # Floating parameters, required for smearing
    shift_mu = zfit.Parameter('delta_mu' + name_tags(tag_list), 0., -200., 200.)
    scale_sigma = zfit.Parameter('scale_sigma' + name_tags(tag_list), 1., 0.1, 10.)

    # main fit parameters, not allowed to float (though we explicitly say which parameters to float later)
    mu = zfit.Parameter('data_mu' + name_tags(tag_list),
                        parameters['mu' + name_tags(tag_list)],
                        floating=False)
    sigma = zfit.Parameter('data_sigma' + name_tags(tag_list),
                           parameters['sigma' + name_tags(tag_list)],
                           floating=False)
    alphal = zfit.Parameter('data_alphal' + name_tags(tag_list),
                            parameters['alphal' + name_tags(tag_list)],
                            floating=False)
    nl = zfit.Parameter('data_nl' + name_tags(tag_list),
                        parameters['nl' + name_tags(tag_list)],
                        floating=False)
    alphar = zfit.Parameter('data_alphar' + name_tags(tag_list),
                            parameters['alphar' + name_tags(tag_list)],
                            floating=False)
    nr = zfit.Parameter('data_nr' + name_tags(tag_list),
                        parameters['nr' + name_tags(tag_list)],
                        floating=False)
    # Additional floating parameter, required for better simulation of an upper power law tail, only for brem 1, 2
    if b_tag == "b_zero:":
        scale_r = zfit.Parameter('sc_r' + name_tags(tag_list), 1., floating=False)
    else:
        scale_r = zfit.Parameter('sc_r' + name_tags(tag_list), 1., 0.1, 10.)

    # Create composed parameters
    mu_shifted = zfit.ComposedParameter("mu_shifted" + name_tags(tag_list),
                                        mu_shifted_fn,
                                        params=[mu, shift_mu])
    sigma_scaled = zfit.ComposedParameter("sigma_scaled" + name_tags(tag_list),
                                          sigma_scaled_fn,
                                          params=[sigma, scale_sigma])
    nr_scaled = zfit.ComposedParameter("nr_scaled" + name_tags(tag_list),
                                       n_scaled_fn,
                                       params=[nr, scale_r])
    alphar_scaled = zfit.ComposedParameter("alphar_scaled" + name_tags(tag_list),
                                           sigma_scaled_fn,   # used as a general multiplication function
                                           params=[alphar, scale_r])

    # Creating model with new scale/shift parameters
    model = zfit.pdf.DoubleCB(obs=obs, mu=mu_shifted, sigma=sigma_scaled,
                              alphal=alphal, nl=nl, alphar=alphar_scaled, nr=nr_scaled)

    # NLL and minimizer
    nll = zfit.loss.UnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)

    # minimization of shift and scale factors
    if b_tag == "b_zero":
        result = minimizer.minimize(nll, params=[mu_shifted, sigma_scaled])
    else:
        result = minimizer.minimize(nll, params=[mu_shifted, sigma_scaled, scale_r])
    final_params = result.params
    errors = result.hesse()   # calculate errors
    print("Result Valid:", result.valid)
    print("Fit converged:", result.converged)
    print(result.params)
    return final_params, model


# todo: could probably use a more generalized plotting function instead of 2 different fns
def plot_data_fit_result(model, ini_model, data, tag_list):  # plotting both initial and finalized models to compare
    b_tag = tag_list[0]
    t_tag = tag_list[1]
    r_tag = tag_list[2]

    lower, upper = obs.limits

    plot_label = "$B \\rightarrow Kee$\n{0}\n{1}".format(b_tag, t_tag)

    h_bin_width = hist_dict["bin_width"]
    h_num_bins = hist_dict["num_bins"]
    h_xmin = hist_dict["x_min"]
    h_xmax = hist_dict["x_max"]
    h_xlabel = hist_dict["x_label"]
    x_var = hist_dict["x_var"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    data_x, _ = np.histogram(data.values, bins=bins)
    data_sum = data_x.sum()
    plt.clf()
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.errorbar(bin_centers, data_x, xerr=16, fmt="ok", label="data")

    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel("Events/32 MeV")

    main_axes.set_xlabel(h_xlabel)
    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    main_axes.plot(x_plot, ini_model.pdf(x_plot) * data_sum * obs.area() / 50, color='xkcd:blue', label="ini model")
    main_axes.plot(x_plot, model.pdf(x_plot) * data_sum * obs.area() / 50, color='xkcd:red', label="fin model")
    main_axes.legend(title=plot_label, loc="best")
    plt.savefig("../Output/data_fit_plot_{0}_{1}_run_{2}.pdf".format(b_tag, t_tag, r_tag))
    plt.close()
