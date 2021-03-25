import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import csv
from scipy.stats import chisquare
import mplhep as hep

from Hist_Settings import hist_dict


# Here are the functions for fitting J/Psi MC, using the fit's shape to fit data (letting some parameters loose)
# and finally getting parameters for smearing from all of this


# formatting data into zfit-compatible format
def format_data(data, obs):
    return zfit.Data.from_numpy(obs, data.to_numpy())


# service function for parameter naming
def name_tags(tags):
    return "_{0}_{1}_{2}".format(tags["brem_cat"], tags["trig_cat"], tags["run_num"])


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

        mu = zfit.Parameter("mu" + name_tags(tags), initial_parameters['mu'],
                            initial_parameters['mu'] - 500., initial_parameters['mu'] + 500.)
        sigma = zfit.Parameter('sigma' + name_tags(tags), initial_parameters['sigma'], 1., 1000.)
        alphal = zfit.Parameter('alphal' + name_tags(tags), initial_parameters['alphal'], 0.001, 50.)
        nl = zfit.Parameter('nl' + name_tags(tags), initial_parameters['nl'], 0.1, 800.)
        alphar = zfit.Parameter('alphar' + name_tags(tags), initial_parameters['alphar'], 0.001, 50.)
        nr = zfit.Parameter('nr' + name_tags(tags), initial_parameters['nr'], 0.1, 800.)
        model = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)

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

def plot_fit_result(models, data, obs, tags, plt_name, p_params=None):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]

    if p_params:
        tags["run_num"] = "run2"
        delta_mu_v = p_params["delta_mu" + name_tags(tags)]["value"]
        delta_mu_err = p_params["delta_mu" + name_tags(tags)]["error"]
        lambd_v = p_params["lambda" + name_tags(tags)]["value"]
        lambd_err = p_params["lambda" + name_tags(tags)]["error"]
        n_bkgr_v = p_params["n_bgr" + name_tags(tags)]["value"]
        n_bkgr_err = p_params["n_bgr" + name_tags(tags)]["error"]
        n_sig_v = p_params["n_signal" + name_tags(tags)]["value"]
        n_sig_err = p_params["n_signal" + name_tags(tags)]["error"]
        sigma_sc_v = p_params["scale_sigma" + name_tags(tags)]["value"]
        sigma_sc_err = p_params["scale_sigma" + name_tags(tags)]["error"]
        if 'sc_r' + name_tags(tags) in p_params.keys():
            s_r_v = p_params['sc_r' + name_tags(tags)]["value"]
            s_r_err = p_params['sc_r' + name_tags(tags)]["error"]
        else:
            s_r_v = 1.
            s_r_err = 0.

        text = f"$\Delta_\mu = {delta_mu_v:.2f} \pm {delta_mu_err:.2f}$\n" \
               f"$\lambda = {lambd_v:.2g} \pm {lambd_err:.2g}$\n" \
               f"$N_b = {n_bkgr_v:.5f} \pm {n_bkgr_err:.2g}$\n" \
               f"$N_s = {n_sig_v:.0f} \pm {n_sig_err:.0f}$\n" \
               f"$s_\sigma = {sigma_sc_v:.3f} \pm {sigma_sc_err:.3f}$\n" \
               f"$s_r = {s_r_v:.3f} \pm {s_r_err:.3f}$"

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
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()

    if data is not None:
        data_x, _ = np.histogram(data.values, bins=bins)
        data_sum = data_x.sum()
        plot_scale = data_sum * obs.area() / h_num_bins

        # main_axes.errorbar(bin_centers, data_x, xerr=h_bin_width / 2, fmt="ok", label=tags["sample"])
        hep.histplot(main_axes.hist(data.values, bins=bins, log=False, facecolor="none"),
                     color="black", yerr=True, histtype="errorbar", label=tags["sample"])

    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel(h_ylabel)
    main_axes.set_xlabel(h_xlabel)

    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    data_bins = np.linspace(lower[-1][0], upper[0][0], num=h_num_bins)

    chisqs = {}
    p_values = {}
    colors = ["b", "k", "r"]
    i = 0
    for model_name, model in models.items():
        if model.is_extended:
            main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / (h_num_bins * model.get_yield()), colors[i],
                           label=model_name)
            # model.get_yield()
            # chisqs[model_name], p_values[model_name] = chisquare(data_x,
            # (model.ext_pdf(data_bins) * obs.area() / h_num_bins))
        else:
            main_axes.plot(x_plot, model.pdf(x_plot) * plot_scale, colors[i], label=model_name)
            # chisqs[model_name], p_values[model_name] = chisquare(data_x,
        i = i + 1  # (model.pdf(data_bins) * plot_scale))
    main_axes.legend(title=plot_label, loc="upper left")

    if not p_params:
        text = ""
    # for model_name in models.keys():
    # add_text = f"\n{model_name} $\chi^2$ = {chisqs[model_name]:.1f}"
    # text += add_text

    plt.text(0.60, 0.60, text, transform=main_axes.transAxes)

    plt.savefig(f"../Output/{plt_name}/{plt_name}_fit_plot_{b_tag}_{t_tag}_{r_tag}.pdf")
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
