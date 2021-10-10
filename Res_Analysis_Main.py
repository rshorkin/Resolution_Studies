import os
import csv

import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
import pandas
import yaml
from rkhq import cuts, parse_name
from Hist_Settings import initial_params_dict

from Service import *
from Fitting import *
import Smearing

# get data, tag it with brem and trigger tags
data = get_data_from_files()  # the result is {1: {"data": df1, "Jpsi_MC: df2, "rare_MC": df3}, 2: {...}}
data = categorize_by_brem(data)  # creates an additional column with brem tags
data = categorize_by_trig(data)  # creates an additional column with trig tags
zfit.run.set_graph_mode(False)


# create a few helper functions
def full_analysis_w_brem(_data):
    obs_dict = {"mee": zfit.Space('brem_P', limits=(0., 60000.)),
                "mKee": zfit.Space('B_plus_M', limits=(4600, 5800))}

    trigger_tags = ['eTOS']
    brem_tags = ["brem_one"]
    query_str = "brem_cat == @brem_tag and trig_cat == @trig_tag"

    option = "mee"
    x_var = "brem_P"
    mass = 3096.9

    obs = obs_dict[option]

    for run_tag, data_run in _data.items():
        jpsi_sample = data_run["Jpsi_MC"]
        data_sample = data_run["data"]
        rare_sample = data_run["rare_MC"]

        models_by_brem = {"data": {}, "Jpsi_MC": {}, "smeared_MC": {}}
        smeared_MC_by_brem = []

        temp_ini_params_dict = {
            "brem_one": {'mu': 700., 'sigma': 500., 'alphal': 0.23, 'nl': 3.5, 'alphar': 0.5, 'nr': 10.},
            "brem_two": {'mu': 2000., 'sigma': 2000., 'alphal': 0.7, 'nl': 3.5, 'alphar': 0.3, 'nr': 3.5}}

        for brem_tag in brem_tags:
            #  initial_params = initial_params_dict[option][brem_tag]
            initial_params = temp_ini_params_dict[brem_tag]
            for trig_tag in trigger_tags:
                print("Working on {0} {1} samples".format(brem_tag, trig_tag))
                models = {}
                tags = {"run_num": str(run_tag), "brem_cat": brem_tag, "trig_cat": trig_tag}

                jpsi_df = jpsi_sample.query(query_str)
                data_df = data_sample.query(query_str)
                rare_df = rare_sample.query(query_str)

                print("MC EVENTS:", len(jpsi_df.index))
                tags["sample"] = "Jpsi_MC"

                hists = {"data hist": data_df['brem_P'], "mc hist": jpsi_df['brem_P']}
                plot_hists(hists, tags["sample"] + "_" + option, tags, 'photon_momentum', (0, 50000))

                # continue
                # plot_histogram(jpsi_df, tags, tags["sample"] + "_" + option)    # test

                print("####==========####\nFitting MC")
                ini_model = create_initial_model(initial_params, obs, tags)
                models["original MC fit"] = ini_model
                _ = initial_fitter(jpsi_df[x_var], ini_model, obs)
                plot_fit_result(models, jpsi_df[x_var], obs, tags, tags["sample"] + "_" + option)  # test

                continue

                tags["sample"] = "data"
                # plot_histogram(data_df, tags, tags["sample"] + "_" + option)   # test

                print("####==========####\nFitting data")
                tags["run_num"] = "run2"
                conv_model, parameters, kernel = convoluted_data_model(ini_model, data_df[x_var], tags, obs)
                models["data fit"] = conv_model
                plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option)  # test
                save_gauss_params(parameters, option, tags)

                continue

                print("####==========####\nSmearing MC")
                tags["sample"] = "Jpsi_MC"
                jpsi_df = Smearing.convolved_smearing(jpsi_df, x_var, kernel=kernel)
                rare_df = Smearing.convolved_smearing(rare_df, x_var, kernel=kernel)

                print("####==========####\nFitting smeared MC")
                tags["run_num"] = str(run_tag) + "_smeared"
                # sm_model = create_initial_model(initial_params, obs, tags)
                # _ = initial_fitter(jpsi_df[x_var + "_smeared"], sm_model, obs)
                # models["smeared MC fit"] = sm_model
                # print('plotting')
                # plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option,
                #                 pulls_switch=True)

                #    migration, sm_migration = Smearing.calc_migration(data=rare_df, cut=19.)
                #    save_gauss_params(migration, option + '_migration', tags)
                #    save_gauss_params(sm_migration, option + '_sm_migration', tags)

                hists = {"data hist": data_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plt_name = 'data vs smeared MC'
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                plt_name = 'data vs MC'
                hists = {"data hist": data_df[x_var], "mc hist": jpsi_df[x_var]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                plt_name = 'MC vs smeared MC'
                hists = {"mc hist": jpsi_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)


def convolve_momentum(data, momentum_var='TRACK_P', csv_smearing=False):
    if 'TRACK_P' in momentum_var:
        option = 'TRACK_P'
    obs_dict = {"brem_PE": zfit.Space(momentum_var, limits=(0., 60000.)),
                "TRACK_PE": zfit.Space(momentum_var, limits=(10000., 250000.)),
                "brem_PZ": zfit.Space(momentum_var, limits=(0., 60000.)),
                "TRACK_P": zfit.Space(momentum_var, limits=(10000., 250000.))}

    trigger_tags = ['eTOS']
    brem_tags = ["brem_one"]
    query_str = "brem_cat == @brem_tag and trig_cat == @trig_tag"

    x_var = momentum_var

    obs = obs_dict[option]

    for run_tag, data_run in data.items():
        jpsi_sample = data_run["Jpsi_MC"]
        data_sample = data_run["data"]
        rare_sample = data_run["rare_MC"]

        for brem_tag in brem_tags:
            initial_params = initial_params_dict[option][brem_tag]
            for trig_tag in trigger_tags:
                print("Working on {0} {1} samples".format(brem_tag, trig_tag))
                models = {}
                tags = {"run_num": str(run_tag), "brem_cat": brem_tag, "trig_cat": trig_tag, 'smeared': ''}

                jpsi_df = jpsi_sample.query(query_str)
                data_df = data_sample.query(query_str)
                rare_df = rare_sample.query(query_str)

                print("MC EVENTS:", len(jpsi_df.index))
                tags["sample"] = "Jpsi_MC"

                hists = {"data hist": data_df[momentum_var], "mc hist": jpsi_df[momentum_var]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, momentum_var)

                # continue
                if not csv_smearing:
                    print("####==========####\nFitting MC")
                    ini_model = create_initial_model(initial_params, obs, tags, momentum_var)
                    models["original MC fit"] = ini_model
                    _ = initial_fitter(jpsi_df[x_var], ini_model, obs)
                    plot_fit_result(models, jpsi_df[x_var], obs, tags, tags["sample"] + "_" + option)  # test

                    # plot_histogram(data_df, tags, tags["sample"] + "_" + option)   # test

                    print("####==========####\nFitting data")
                    tags["sample"] = "data"
                    conv_model, parameters, kernel = convoluted_data_model(ini_model, data_df[x_var],
                                                                           tags, obs, momentum_var)
                    models["data fit"] = conv_model
                    plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option)  # test
                    save_gauss_params(parameters, option, tags)

                    print("####==========####\nSmearing MC")
                    tags["sample"] = "Jpsi_MC"
                    jpsi_df = Smearing.convolved_smearing(jpsi_df, x_var, kernel=kernel)
                    # rare_df = Smearing.convolved_smearing(rare_df, x_var, kernel=kernel)
                else:
                    jpsi_df = Smearing.smear_from_csv(jpsi_df, x_var, tags, option)

                print("####==========####\nFitting smeared MC")
                # sm_model = create_initial_model(initial_params, obs, tags)
                # _ = initial_fitter(jpsi_df[x_var + "_smeared"], sm_model, obs)
                # models["smeared MC fit"] = sm_model
                # plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option,
                #                 pulls_switch=True)

                hists = {"data hist": data_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plt_name = 'data vs smeared MC'
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                plt_name = 'data vs MC'
                hists = {"data hist": data_df[x_var], "mc hist": jpsi_df[x_var]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                plt_name = 'MC vs smeared MC'
                hists = {"mc hist": jpsi_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

    return jpsi_df, data_df


def plot_hists(dict, plt_name, tags, plt_title):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]
    plt.figure()
    h_num_bins = hist_dict[plt_name]["num_bins"]
    h_xmin = hist_dict[plt_name]["x_min"]
    # h_num_bins = 100
    # h_xmin = 0
    h_xmax = hist_dict[plt_name]["x_max"]
    # h_xmax = 4000
    h_xlabel = hist_dict[plt_name]["x_label"]
    h_ylabel = hist_dict[plt_name]["y_label"]
    for key, value in dict.items():
        plt.hist(value, bins=h_num_bins, alpha=0.5, range=(h_xmin, h_xmax), density=True, label=key)
        # plt.hist(value, bins=100, alpha=0.5, density=True, label=key, range=range)

    plt.legend(loc='upper left')
    plt.title(plt_title)
    if not os.path.exists(f'../Results/Hists/{plt_name}_{plt_title}'):
        os.makedirs(f'../Results/Hists/{plt_name}_{plt_title}')
    plt.savefig(f'../Results/Hists/{plt_name}_{plt_title}/{b_tag}_{t_tag}_{r_tag}_{plt_title}_hist.jpg')
    plt.close()


def save_gauss_params(params, naming, tags):
    r_tag = tags["run_num"]
    b_tag = tags["brem_cat"]
    t_tag = tags["trig_cat"]
    # if not os.path.exists(f'../Results/Parameters/{naming}/{b_tag}_{t_tag}_{r_tag}'):
    #    os.makedirs(f'../Results/Parameters/{naming}/{b_tag}_{t_tag}_{r_tag}')
    if not os.path.exists(f'../Results/Parameters/{naming}'):
        os.makedirs(f'../Results/Parameters/{naming}')

    a_file = open(f'../Results/Parameters/{naming}/{b_tag}_{t_tag}_{r_tag}.csv', "w")
    writer = csv.writer(a_file)
    for key, value in params.items():
        writer.writerow([key, value])
    a_file.close()
    return None


# br_jpsi_df, _ = convolve_momentum(data, momentum_var='brem_P')
trpl_jpsi_df, _ = convolve_momentum(data, momentum_var='e_plus_TRACK_P', csv_smearing=True)
trmin_jpsi_df, _ = convolve_momentum(data, momentum_var='e_minus_TRACK_P', csv_smearing=True)

trmin_jpsi_df['q2_nobrem_smeared'] = (trpl_jpsi_df['TRACK_PE'] ** 2 - \
                                      (trpl_jpsi_df['e_plus_TRACK_P_smeared'] ** 2 + trmin_jpsi_df[
                                          'e_minus_TRACK_P_smeared'] ** 2 +
                                       2 * trpl_jpsi_df['e_plus_TRACK_P_smeared'] *
                                       trmin_jpsi_df['e_minus_TRACK_P_smeared'] *
                                       trmin_jpsi_df['cosTheta'])) / 10 ** 6

hists = {"data hist": trmin_jpsi_df['q2_nobrem'], "smeared mc hist": trmin_jpsi_df['q2_nobrem_smeared']}
plt_name = 'data vs smeared MC'
for key, value in hists.items():
    plt.clf()
    plt.hist(value, bins=100, alpha=0.5, density=True, label=key)
    plt.show()
