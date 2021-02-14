import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator

from Hist_Settings import hist_dict

# Here are the functions for fitting J/Psi MC, using the fit's shape to fit data (letting some parameters loose)
# and finally getting parameters for smearing from all of this

# todo: generalize functions to work with both m(ee) and m(Kee) (different obs, rework format_data(data))
# todo: use extended pdfs instead of pdfs to produced composed model incorporating backgrounds
# formatting data into zfit-compatible format


def format_data(data, obs):
    return zfit.Data.from_numpy(obs, data.to_numpy())


# service function for parameter naming
def name_tags(tags):
    return "_{0}_{1}_{2}".format(tags["brem_cat"], tags["trig_cat"], tags["run_num"])


# Creating initial model to fit J/Psi
def create_initial_model(initial_parameters, obs, tags):
    # Double Crystal Ball for each category, like in RK

    mu = zfit.Parameter("mu" + name_tags(tags), initial_parameters['mu'],
                        initial_parameters['mu'] - 100., initial_parameters['mu'] + 100.)
    sigma = zfit.Parameter('sigma' + name_tags(tags), initial_parameters['sigma'], 1., 100.)
    alphal = zfit.Parameter('alphal' + name_tags(tags), initial_parameters['alphal'], 0., 5.)
    nl = zfit.Parameter('nl' + name_tags(tags), initial_parameters['nl'], 0., 200.)
    alphar = zfit.Parameter('alphar' + name_tags(tags), initial_parameters['alphar'], 0., 5.)
    nr = zfit.Parameter('nr' + name_tags(tags), initial_parameters['nr'], 0., 200.)

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
        print(f'>>> Result is valid')
        print("Converged:", result.converged)
        param_errors = result.hesse()
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

def plot_fit_result(models, data, obs, tags, plt_name):
    r_tag = tags["run_num"]
    if "brem_cat" in tags:
        b_tag = tags["brem_cat"]
    else:
        b_tag = "all brem cats"
    if "trig_cat" in tags:
        t_tag = tags["trig_cat"]
    else:
        t_tag = "all trig cats"

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

    data_x, _ = np.histogram(data.values, bins=bins)
    data_sum = data_x.sum()
    plot_scale = data_sum * obs.area() / h_num_bins

    plt.clf()
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.errorbar(bin_centers, data_x, xerr=h_bin_width / 2, fmt="ok", label=tags["sample"])

    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel(h_ylabel)
    main_axes.set_xlabel(h_xlabel)

    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    for model_name, model in models.items():
        main_axes.plot(x_plot, model.pdf(x_plot) * plot_scale, label=model_name)
    main_axes.legend(title=plot_label, loc="best")
    plt.savefig("../Output/{0}_fit_plot_{1}_{2}_run_{3}.pdf".format(plt_name, b_tag, t_tag, r_tag))
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
def create_data_fit_model(data, parameters, obs, tags):
    if "brem_cat" in tags:
        b_tag = tags["brem_cat"]
    else:
        b_tag = "all_brem"
    num_events = len(data.index)
    data = format_data(data, obs)
    # Initializing new model parameters to save the previous model for comparison
    # Floating parameters, required for smearing
    shift_mu = zfit.Parameter('delta_mu' + name_tags(tags), 0., -100., 100.)
    scale_sigma = zfit.Parameter('scale_sigma' + name_tags(tags), 1., 0.1, 5.)

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
    if b_tag == "b_zero:":
        scale_r = zfit.Parameter('sc_r' + name_tags(tags), 1., floating=False)
    else:
        scale_r = zfit.Parameter('sc_r' + name_tags(tags), 1., 0.1, 10.)

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
    lambd = zfit.Parameter("lambda" + name_tags(tags), -0.005, -0.1, 0.1)
    model_bgr = zfit.pdf.Exponential(lambd, obs=obs)

    # Make models extended and combine them
    n_sig = zfit.Parameter("n_signal" + name_tags(tags),
                           int(num_events * 0.999), int(num_events * 0.7), int(num_events * 1.1), step_size=1)
    n_bgr = zfit.Parameter("n_bgr" + name_tags(tags),
                           int(num_events * 0.001), 0, int(num_events * 0.3), step_size=1)

    model_extended = model.create_extended(n_sig)
    model_bgr_extended = model_bgr.create_extended(n_bgr)

    model = zfit.pdf.SumPDF([model_extended, model_bgr_extended])
    # NLL and minimizer
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True)

    # minimization of shift and scale factors
    if b_tag == "b_zero":
        result = minimizer.minimize(nll, params=[lambd, n_sig, n_bgr, mu_shifted, sigma_scaled])
    else:
        result = minimizer.minimize(nll, params=[lambd, n_sig, n_bgr, mu_shifted, sigma_scaled, scale_r])
    final_params = result.params
    param_errors = result.hesse()
    print("Result Valid:", result.valid)
    print("Fit converged:", result.converged)
    print(result.params)
    return {param[0].name: {"value": param[1]['value'], "error": err[1]['error']}
            for param, err in zip(result.params.items(), param_errors.items())}, model
