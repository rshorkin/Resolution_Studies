import scipy.stats
import zfit
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, LogFormatterSciNotation, NullFormatter
import mplhep as hep
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


def plot_fit_result(models, data, obs, tags, plt_name, pulls_switch=False):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]
    if tags['smeared'] != '':
        tags["run_num"] = tags["run_num"] + '_' + tags['smeared']

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

    if not os.path.exists(f'../Results/Plots/{plt_name}/{b_tag}_{t_tag}_{r_tag}'):
        os.makedirs(f'../Results/Plots/{plt_name}/{b_tag}_{t_tag}_{r_tag}')
    plt.savefig(f'../Results/Plots/{plt_name}/{b_tag}_{t_tag}_{r_tag}/{plt_name}_fit_plot_{b_tag}_{t_tag}_{r_tag}.jpg')
    print(
        'saved ' + f'../Results/Plots/{plt_name}/{b_tag}_{t_tag}_{r_tag}/{plt_name}_fit_plot_{b_tag}_{t_tag}_{r_tag}.jpg')
    # plt.show()
    plt.close()


def plot_hists(hists: dict, tags, bin_range, nbins=100, save_file='', xlabel='', ylabel='', title='', stand_hist=None,
               log=False, x_log=False, loc='right', weights=None):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]
    if tags['smeared'] != '':
        tags["run_num"] = tags["run_num"] + '_' + tags['smeared']

    plot_label = ''
    h_bin_width = (bin_range[1] - bin_range[0]) / nbins
    h_num_bins = nbins
    h_xmin = bin_range[0]
    h_xmax = bin_range[1]
    h_xlabel = xlabel
    h_ylabel = ylabel
    title = title

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]
    if x_log:
        logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
        bin_centers = [(logbins[i] + logbins[i + 1]) / 2 for i in range(len(logbins) - 1)]

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    main_axes.set_title(title, fontsize=18)

    pulls_dict = {}
    ratio_err_dict = {}
    stand_err_dict = {}

    for key, value in hists.items():
        if not x_log:
            if weights:
                data_x, _ = np.histogram(value, bins=h_num_bins, range=bin_range, weights=weights[key])
            else:
                data_x, _ = np.histogram(value, bins=h_num_bins, range=bin_range)
        else:
            if weights:
                data_x, _ = np.histogram(value, bins=logbins, weights=weights[key])
            else:
                data_x, _ = np.histogram(value, bins=logbins)
        data_errors = np.sqrt(data_x)
        data_sum = data_x.sum()

        data_x = data_x / data_sum
        data_errors = data_errors / data_sum

        main_axes.errorbar(bin_centers, data_x, yerr=data_errors, label=f'{key}', fmt='o',
                           markersize='4')

        if stand_hist:
            if stand_hist != key:
                if not x_log:
                    stand_x, _ = np.histogram(hists[stand_hist], bins=h_num_bins, range=bin_range)
                else:
                    stand_x, _ = np.histogram(hists[stand_hist], bins=logbins)
                stand_errors = np.sqrt(stand_x)
                stand_sum = stand_x.sum()

                stand_x = stand_x / stand_sum
                stand_errors = stand_errors / stand_sum

                cond = np.not_equal(stand_x, 0.)
                pulls_dict[key] = np.where(cond,
                                           np.divide(
                                               data_x,
                                               stand_x),
                                           1.)

                cond = np.not_equal(data_x, 0.)
                ratio_err_dict[key] = np.where(cond,
                                               np.divide(stand_errors, data_x),
                                               0.)

                cond = np.not_equal(stand_x, 0.)
                stand_err_dict[key] = np.where(cond,
                                               np.divide(data_errors, stand_x),
                                               0.)

    main_axes.set_xlim(h_xmin, h_xmax)
    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_ylabel(h_ylabel)

    main_axes.legend(title=plot_label, loc=f"upper {loc}")
    main_axes.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))

    if log:
        main_axes.set_yscale('log')
        bottom = min([min(data_x) * 0.7, 10.])
        main_axes.set_ylim(bottom=bottom)
        main_axes.yaxis.set_major_formatter(CustomTicker())
        locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
        main_axes.yaxis.set_minor_locator(locmin)
    if not log:
        main_axes.set_ylim(bottom=0)

    if x_log:
        main_axes.set_xscale('log')

    if stand_hist is None:
        main_axes.set_xlabel(h_xlabel)

    if stand_hist:
        main_axes.set_xticklabels([])
        # pulls subplot
        plt.axes([0.1, 0.1, 0.85, 0.2])
        plt.yscale("linear")
        pulls_axes = plt.gca()
        for key in hists.keys():
            if key != stand_hist:
                pulls = pulls_dict[key]
                if not x_log:
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
                    pulls_axes.plot(xs, ys, label=f'"Ratio" for {key}')
                else:
                    xs = [h_xmin]
                    ys = [pulls[0]]
                    for i in range(h_num_bins - 1):
                        xs.append(logbins[i + 1])
                        xs.append(logbins[i + 1])
                        ys.append(pulls[i])
                        ys.append(pulls[i + 1])
                    del i
                    xs.append(h_xmax)
                    ys.append(pulls[-1])
                    pulls_axes.plot(xs, ys, label=f'"Ratio" for {key}')

                if not x_log:
                    pulls_axes.bar(bin_centers, 2 * ratio_err_dict[key],
                                   bottom=pulls - ratio_err_dict[key],
                                   alpha=0.5, color='blue',
                                   hatch=r"\\\\", width=h_bin_width)
                    pulls_axes.bar(bin_centers, 2 * stand_err_dict[key],
                                   bottom=np.ones_like(stand_err_dict[key]) - stand_err_dict[key],
                                   alpha=0.5, color='none',
                                   hatch="////", width=h_bin_width)
                else:
                    widths = [logbins[i + 1] - logbins[1] for i in range(len(logbins) - 1)]
                    pulls_axes.bar(bin_centers, 2 * ratio_err_dict[key],
                                   bottom=pulls - ratio_err_dict[key],
                                   alpha=0.5, color='blue',
                                   hatch=r"\\\\", width=widths)
                    pulls_axes.bar(bin_centers, 2 * stand_err_dict[key],
                                   bottom=np.ones_like(stand_err_dict[key]) - stand_err_dict[key],
                                   alpha=0.5, color='none',
                                   hatch="////", width=widths)

        pulls_axes.set_yscale('log')
        # locmin = LogLocator(base=10.0, subs=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9), numticks=12)
        # pulls_axes.yaxis.set_minor_locator(locmin)

        if x_log:
            pulls_axes.set_xscale('log')

        pulls_axes.set_ylim(0.3, 3.)
        pulls_axes.set_yticks([0.5, 1., 2.])
        pulls_axes.yaxis.set_minor_formatter(NullFormatter())
        pulls_axes.set_yticklabels([0.5, 1., 2.])

        pulls_axes.set_xlabel(h_xlabel)
        pulls_axes.set_ylabel('Ratio')
        pulls_axes.set_xlim(h_xmin, h_xmax)
        plt.grid("True", axis="y", color="black", linestyle="--")

    if not os.path.exists(f'../Results/Hists/{save_file}/'):
        os.makedirs(f'../Results/Hists/{save_file}/')
    plt.savefig(f'../Results/Hists/{save_file}/{title}_hist_{b_tag}_{t_tag}_{r_tag}.jpg')
    print('saved ' + f'../Results/Hists/{save_file}/{title}_hist_{b_tag}_{t_tag}_{r_tag}.jpg')
    # plt.show()
    plt.close()


def plot_2dhist(data_x, data_y, weights=None, bins=None, range=None,
                title='', xlabel='', yalbel='', clabel='', filename='', path='../Results/Hists2d/General'):
    plt.clf()
    fig, ax = plt.subplots()
    h, xedges, yedges, image = ax.hist2d(data_x, data_y, weights=weights, bins=bins, range=range, cmin=1)
    # hxy, _, _ = np.histogram2d(sig_cut_df.xmean, sig_cut_df.ymean, bins=300, range=[[-75., 75.], [-75., 75]])
    # hxy_pmt0 = np.divide(hxy_pmt0, hxy)
    # hxy = np.transpose(hxy)
    pearson_r = scipy.stats.pearsonr(data_x, data_y)[0]
    spearman_r = scipy.stats.spearmanr(data_x, data_y)[0]
    kendall_t = scipy.stats.kendalltau(data_x, data_y)[0]

    plt.text(0.77, 0.99, f"Entries: {len(data_x.index)}\nPearson's r: {pearson_r:.2f}\nSpearman's rho: {spearman_r:.2f}"
                         f"\nKendall's tau: {kendall_t:.2f}",
             ha="left", va="top", family='sans-serif',
             fontsize=10, transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    # h = ax.pcolor(ys, xs, hxy, vmin=1.)

    fig.colorbar(image, ax=ax, label=clabel)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(yalbel)
    if not os.path.exists(f'{path}'):
        os.makedirs(f'{path}')
    plt.savefig(f'{path}/{filename}.jpg')
    plt.clf()