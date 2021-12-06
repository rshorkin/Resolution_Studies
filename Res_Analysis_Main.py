import os
import csv

# e_plus_L0Calo_ECAL_region


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
from Plotting import plot_hists, plot_fit_result, plot_2dhist

# get data, tag it with brem and trigger tags
data = get_data_from_files()  # the result is {1: {"data": df1, "Jpsi_MC: df2, "rare_MC": df3}, 2: {...}}
data = categorize_by_brem(data)  # creates an additional column with brem tags
data = categorize_by_trig(data)  # creates an additional column with trig tags
zfit.run.set_graph_mode(False)


# create a few helper functions
def full_analysis_w_brem(_data):
    obs_dict = {"mee": zfit.Space('J_psi_1S_M', limits=(2400, 3600)),
                "mKee": zfit.Space('B_plus_M', limits=(4600, 5800))}

    data_select = int(input("Choose the type of analysis\n0 for all trigger and brem categories\n"
                            "1 for all brem categories (trigger independent)\n"))
    if data_select == 0:
        trigger_tags = ["TIS", 'eTOS']
        brem_tags = ["brem_one", "brem_two"]
        query_str = "brem_cat == @brem_tag and trig_cat == @trig_tag"
    elif data_select == 1:
        trigger_tags = ["TIS_or_eTOS"]
        brem_tags = ["brem_zero", "brem_one", "brem_two"]
        query_str = "brem_cat == @brem_tag"
    else:
        raise ValueError("Error! This choice is currently not supported. Please select a number from 0 to 2")

    switch = int(input("Please select between m(ee) and m(Kee) analysis\n0 for m(ee), 1 for m(Kee)\n"))
    if switch == 0:
        option = "mee"
        x_var = "J_psi_1S_M"
        mass = 3096.9
    elif switch == 1:
        option = "mKee"
        x_var = "B_plus_M"
        mass = 5279.26
    else:
        raise ValueError("Error! Choice is not in (0, 1)!")
    obs = obs_dict[option]

    read_switch = int(input("0 to calculate parameters, 1 to read them from csv\n"))
    for run_tag, data_run in _data.items():
        jpsi_sample = data_run["Jpsi_MC"]
        data_sample = data_run["data"]
       # rare_sample = data_run["rare_MC"]

        models_by_brem = {"data": {}, "Jpsi_MC": {}, "smeared_MC": {}}
        smeared_MC_by_brem = []

        for brem_tag in brem_tags:
            initial_params = initial_params_dict[option][brem_tag]
            for trig_tag in trigger_tags:
                print("Working on {0} {1} samples".format(brem_tag, trig_tag))
                models = {}
                tags = {"run_num": str(run_tag), "brem_cat": brem_tag, "trig_cat": trig_tag}

                jpsi_df = jpsi_sample.query(query_str)
                data_df = data_sample.query(query_str)
            #    rare_df = rare_sample.query(query_str)
                print("MC EVENTS:", len(jpsi_df.index))
                tags["sample"] = "Jpsi_MC"
                # plot_histogram(jpsi_df, tags, tags["sample"] + "_" + option)    # test
                if read_switch == 0:
                    print("####==========####\nFitting MC")
                    ini_model = create_initial_model(initial_params, obs, tags)
                    models["original MC fit"] = ini_model
                    mc_fit_params = initial_fitter(jpsi_df[x_var], ini_model, obs)
                    # plot_fit_result(models, jpsi_df[x_var], obs, tags, tags["sample"] + "_" + option)  # test

                    tags["sample"] = "data"
                    # plot_histogram(data_df, tags, tags["sample"] + "_" + option)   # test

                    print("####==========####\nFitting data")
                    tags["run_num"] = "run2"
                    conv_model, parameters, kernel = convoluted_data_model(ini_model, data_df[x_var], tags, obs,
                                                                           nobrem=False)
                    models["data fit"] = conv_model
                    # plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option, pulls_switch=True)
                    save_gauss_params(parameters, option, tags)

                    print("####==========####\nSmearing MC")
                    tags["sample"] = "Jpsi_MC"
                    jpsi_df = Smearing.convolved_smearing(jpsi_df, x_var, kernel=kernel)
                #    rare_df = Smearing.convolved_smearing(rare_df, x_var, kernel=kernel)
                elif read_switch == 1:
                    jpsi_df = Smearing.smear_from_csv(data=jpsi_df, naming='mee_nobrem', tags=tags, x_var=x_var)
               #     rare_df = Smearing.smear_from_csv(data=rare_df, naming='mee_nobrem', tags=tags, x_var=x_var,
               #                                       add='_rare')
                print("####==========####\nFitting smeared MC")
                tags["run_num"] = str(run_tag) + "_smeared"
                sm_model = create_initial_model(initial_params, obs, tags)
                _ = initial_fitter(jpsi_df[x_var + "_smeared"], sm_model, obs)
                models["smeared MC fit"] = sm_model
                print('plotting')
                plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option,
                                pulls_switch=True)

            #    migration, sm_migration = Smearing.calc_migration(data=rare_df, cut=19.)
            #    save_gauss_params(migration, option + '_migration', tags)
            #    save_gauss_params(sm_migration, option + '_sm_migration', tags)
                if data_select == 1:
                    models_by_brem["data"][brem_tag] = models["data fit"]
                    models_by_brem["Jpsi_MC"][brem_tag] = models["original MC fit"]
                    models_by_brem["smeared_MC"][brem_tag] = models["smeared MC fit"]
                    smeared_MC_by_brem.append(jpsi_df)

                hists = {"data hist": data_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plt_name = 'data vs smeared MC'
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                plt_name = 'data vs MC'
                hists = {"data hist": data_df[x_var], "mc hist": jpsi_df[x_var]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                plt_name = 'MC vs smeared MC'
                hists = {"mc hist": jpsi_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

        if data_select == 1:
            full_models = {}
            sMC_sample = pandas.concat(smeared_MC_by_brem)
            tags = {"run_num": str(run_tag), "sample": "Jpsi_MC", "trig_cat": "TIS_&_eTOS", "brem_cat": ""}

            mc_model = combine_models(models_by_brem["Jpsi_MC"], jpsi_sample[x_var], obs, tags)
            full_models["initial MC fit"] = mc_model

            tags["sample"] = "data"
            data_model = combine_models(models_by_brem["data"], data_sample[x_var], obs, tags)
            full_models["data fit"] = data_model

            tags["sample"] = "smeared_MC"
            sMC_model = combine_models(models_by_brem["smeared_MC"], sMC_sample[x_var + "_smeared"], obs, tags)
            full_models["smeared MC fit"] = sMC_model

            plot_fit_result(full_models, None, obs, tags, "data" + "_" + option, pulls_switch=False)


def full_analysis_no_brem(_data):
    obs = zfit.Space('mee_nobrem', limits=(300, 3300))
    models = {}
    option = "mee_nobrem"
    x_var = "J_psi_1S_TRACK_M"
    mass = 3096.9
    read_switch = 1

    data_select = int(input("Choose the type of analysis\n0 for all trigger and brem categories\n"
                            "1 for all brem categories (trigger independent)\n"))
    if data_select == 0:
        trigger_tags = ['TIS', 'eTOS']
        brem_tags = ['brem_one', 'brem_two']
        query_str = "brem_cat == @brem_tag and trig_cat == @trig_tag"
    elif data_select == 1:
        trigger_tags = ["TIS_or_eTOS"]
        brem_tags = ['brem_one', "brem_two"]
        query_str = "brem_cat == @brem_tag"
    for run_tag, data_run in _data.items():

        jpsi_sample = data_run["Jpsi_MC"]
        data_sample = data_run["data"]
    #    rare_sample = data_run["rare_MC"]

        models_by_brem = {"data": {}, "Jpsi_MC": {}, "smeared_MC": {}}
        smeared_MC_by_brem = []
        for brem_tag in brem_tags:
            initial_params = initial_params_dict[option][brem_tag]
            for trig_tag in trigger_tags:
                tags = {"run_num": str(run_tag), "brem_cat": brem_tag, "trig_cat": trig_tag}
                models = {}
                jpsi_df = jpsi_sample.query(query_str)
                data_df = data_sample.query(query_str)
            #    rare_df = rare_sample.query(query_str)
                print("MC EVENTS:", len(jpsi_df.index))
                tags["sample"] = "Jpsi_MC"
                if read_switch == 0:
                    if brem_tag != "brem_zero":
                        ini_model = create_initial_model(initial_params, obs, tags, switch='nobrem')
                    else:
                        ini_model = create_initial_model(initial_params, obs, tags)
                    models["original MC fit"] = ini_model
                    mc_fit_params = initial_fitter(jpsi_df[x_var], ini_model, obs)
                    plot_fit_result(models, jpsi_df[x_var], obs, tags, tags["sample"] + "_" + option,
                                    pulls_switch=True)  # test

                    print('"####==========####\nFitting data"')

                    tags["sample"] = "data"
                    conv_model, parameters, kernel = convoluted_data_model(ini_model, data_df[x_var], tags, obs,
                                                                           nobrem=True)
                    models["data fit"] = conv_model
                    plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option, pulls_switch=True)

                    print('"####==========####\nSmearing J/Psi MC"')
                    tags["sample"] = "Jpsi_MC"
                    jpsi_df = Smearing.convolved_smearing(jpsi_df, x_var, kernel=kernel)
                #    rare_df = Smearing.convolved_smearing(rare_df, x_var, kernel=kernel)
                    save_gauss_params(parameters, option, tags)
                elif read_switch == 1:
                    jpsi_df = Smearing.smear_from_csv(data=jpsi_df, naming='mee_nobrem', tags=tags, x_var=x_var)
                #    rare_df = Smearing.smear_from_csv(data=rare_df, naming='mee_nobrem', tags=tags, x_var=x_var,
                #                                      add='_rare')

                plt_name = 'data vs smeared MC'
                hists = {"data hist": data_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                plt_name = 'data vs MC'
                hists = {"data hist": data_df[x_var], "mc hist": jpsi_df[x_var]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                plt_name = 'MC vs smeared MC'
                hists = {"mc hist": jpsi_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plot_hists(hists, tags["sample"] + "_" + option, tags, plt_name)

                print("####==========####\nFitting smeared MC")
                tags["run_num"] = str(run_tag) + "_smeared"
                if brem_tag != "brem_zero":
                    sm_model = create_initial_model(initial_params, obs, tags, switch='nobrem')
                else:
                    sm_model = create_initial_model(initial_params, obs, tags)
                models["smeared MC fit"] = sm_model
                _ = initial_fitter(jpsi_df[x_var + "_smeared"], sm_model, obs)

                plot_fit_result(models, jpsi_df[x_var + "_smeared"], obs, tags, tags["sample"] + "_" + option,
                                pulls_switch=True)

            #    migration, sm_migration = Smearing.calc_migration(data=rare_df, cut=14.3, nobrem=True, true_cut=14.3)
            #    save_gauss_params(migration, option + '_migration', tags)
            #    save_gauss_params(sm_migration, option + '_sm_migration', tags)


def smear_q2_TRACK(_data):
    obs = zfit.Space('mee_nobrem', limits=(300, 3300))
    option = "mee_nobrem"
    x_var = "J_psi_1S_TRACK_M"
    read_switch = 1

    trigger_tags = ['TIS', 'eTOS']
    brem_tags = ['brem_one', 'brem_two']
    query_str = "brem_cat == @brem_tag and trig_cat == @trig_tag"
 #   trigger_tags = ['all_trig']
 #   brem_tags = ['all_brem']
    #query_str = "brem_cat == @brem_tag and trig_cat == @trig_tag"
    for run_tag, data_run in _data.items():

        jpsi_sample = data_run["Jpsi_MC"]
        data_sample = data_run["data"]
    #    rare_sample = data_run["rare_MC"]

        for brem_tag in brem_tags:
            initial_params = initial_params_dict[option][brem_tag]
            for trig_tag in trigger_tags:
                tags = {"run_num": str(run_tag), "brem_cat": brem_tag, "trig_cat": trig_tag, 'smeared': ''}
                models = {}
                jpsi_df = jpsi_sample.query(query_str)
             #   jpsi_df = jpsi_sample
                data_df = data_sample.query(query_str)
            #    data_df = data_sample
            #    rare_df = rare_sample.query(query_str)
                print("MC EVENTS:", len(jpsi_df.index))
                tags["sample"] = "Jpsi_MC"
                if read_switch == 0:
                    if brem_tag != "brem_zero":
                        ini_model = create_initial_model(initial_params, obs, tags, switch='nobrem')
                    else:
                        ini_model = create_initial_model(initial_params, obs, tags)
                    models["original MC fit"] = ini_model
                    mc_fit_params = initial_fitter(jpsi_df[x_var], ini_model, obs)
                    plot_fit_result(models, jpsi_df[x_var], obs, tags, tags["sample"] + "_" + option,
                                    pulls_switch=True)  # test

                    print('"####==========####\nFitting data"')

                    tags["sample"] = "data"
                    conv_model, parameters, kernel = convoluted_data_model(ini_model, data_df[x_var], tags, obs,
                                                                           nobrem=True)
                    models["data fit"] = conv_model
                    plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option, pulls_switch=True)

                    print('"####==========####\nSmearing J/Psi MC"')
                    tags["sample"] = "Jpsi_MC"
                    jpsi_df = Smearing.convolved_smearing(jpsi_df, x_var, kernel=kernel)
                    #    rare_df = Smearing.convolved_smearing(rare_df, x_var, kernel=kernel)
                    save_gauss_params(parameters, option, tags)
                elif read_switch == 1:
                    jpsi_df = Smearing.smear_from_csv(data=jpsi_df, naming='mee_nobrem', tags=tags, x_var=x_var)
                    # rare_df = Smearing.smear_from_csv(data=rare_df, naming='mee_nobrem', tags=tags,
                    # x_var=x_var, add='_rare')

                # =================================================================================

                plt_name = 'data vs smeared MC'
                hists = {"data hist": data_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plot_hists(hists, tags=tags, nbins=100, bin_range=(300, 3300),
                           save_file=x_var,
                           xlabel=r'$J/\psi TRACK mass, [MeV/c^2]$',
                           ylabel='Normed events / 30',
                           title=plt_name,
                           stand_hist="data hist")

                plt_name = 'data vs MC'
                hists = {"data hist": data_df[x_var], "mc hist": jpsi_df[x_var]}
                plot_hists(hists, tags=tags, nbins=100, bin_range=(300, 3300),
                           save_file=x_var,
                           xlabel=r'$J/\psi TRACK mass, [MeV/c^2]$',
                           ylabel='Normed events / 30',
                           title=plt_name,
                           stand_hist="data hist")

                plt_name = 'MC vs smeared MC'
                hists = {"mc hist": jpsi_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plot_hists(hists, tags=tags, nbins=100, bin_range=(300, 3300),
                           save_file=x_var,
                           xlabel=r'$J/\psi$ TRACK mass, $[MeV/c^2]$',
                           ylabel='Normed events / 30',
                           title=plt_name,
                           stand_hist="mc hist")

                # =================================================================================

                print("####==========####\nSmearing q2")
                jpsi_df = smear_TRACK_P(jpsi_df)
                jpsi_df = calc_smeared_q2(jpsi_df)

                plt_name = 'MC smeared hist'
                hists = {"mc hist": jpsi_df['smear_factor']}
                plot_hists(hists, tags=tags, nbins=40, bin_range=(0.9, 1.1),
                           save_file='MC smear factor',
                           xlabel=r'smear factor values',
                           ylabel='Normed events',
                           title=plt_name,
                           stand_hist=None)

                plt_name = 'data vs smeared MC q2'
                hists = {"data hist": data_df['q2'], "smeared mc hist": jpsi_df["q2_smeared"]}
                plot_hists(hists, tags=tags, nbins=35, bin_range=(6, 13),
                           save_file='q2 assume factor',
                           xlabel=r'$q^2, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist")

                plt_name = 'data vs MC q2'
                hists = {"data hist": data_df['q2'], "mc hist": jpsi_df['q2']}
                plot_hists(hists, tags=tags, nbins=35, bin_range=(6, 13),
                           save_file='q2 assume factor',
                           xlabel=r'$q^2, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist")

                plt_name = 'MC vs smeared MC q2'
                hists = {"mc hist": jpsi_df['q2'], "smeared mc hist": jpsi_df["q2_smeared"]}
                plot_hists(hists, tags=tags, nbins=35, bin_range=(6, 13),
                           save_file='q2 assume factor',
                           xlabel=r'$q^2, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="mc hist")

                # =================================================================================

                plt_name = 'data vs half-smeared MC q2'
                hists = {"data hist": data_df['q2'], "smeared mc hist": jpsi_df["q2_halfsmeared"]}
                plot_hists(hists, tags=tags, nbins=35, bin_range=(6, 13),
                           save_file='q2 smear TRACK only',
                           xlabel=r'$q^2, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist")

                plt_name = 'data vs MC q2'
                hists = {"data hist": data_df['q2'], "mc hist": jpsi_df['q2']}
                plot_hists(hists, tags=tags, nbins=35, bin_range=(6, 13),
                           save_file='q2 smear TRACK only',
                           xlabel=r'$q^2, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist")

                plt_name = 'MC vs half-smeared MC q2'
                hists = {"mc hist": jpsi_df['q2'], "smeared mc hist": jpsi_df["q2_halfsmeared"]}
                plot_hists(hists, tags=tags, nbins=35, bin_range=(6, 13),
                           save_file='q2 smear TRACK only',
                           xlabel=r'$q^2, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="mc hist")

                # =================================================================================

                plt_name = 'data vs smeared MC q2 TRACK'
                hists = {"data hist": data_df['q2_nobrem'], "smeared mc hist": jpsi_df["q2_nobrem_smeared"]}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 10),
                           save_file='q2_TRACK',
                           xlabel=r'$q^2 TRACK, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist",
                           loc='left')

                plt_name = 'data vs MC q2 TRACK'
                hists = {"data hist": data_df['q2_nobrem'], "mc hist": jpsi_df['q2_nobrem']}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 10),
                           save_file='q2_TRACK',
                           xlabel=r'$q^2 TRACK, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist",
                           loc='left')

                plt_name = 'MC vs smeared MC q2 TRACK'
                hists = {"mc hist": jpsi_df['q2_nobrem'], "smeared mc hist": jpsi_df["q2_nobrem_smeared"]}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 10),
                           save_file='q2_TRACK',
                           xlabel=r'$q^2 TRACK, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="mc hist",
                           loc='left')

                # =================================================================================

                plt_name = 'data vs smeared MC q2 ADD'
                hists = {"data hist": data_df['q2_ADD'], "mc hist": jpsi_df["q2_ADD_smeared"]}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 10),
                           save_file='q2_ADD',
                           xlabel=r'$q^2 ADD, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist")

                plt_name = 'data vs MC q2 ADD'
                hists = {"data hist": data_df['q2_ADD'], "mc hist": jpsi_df["q2_ADD"]}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 10),
                           save_file='q2_ADD',
                           xlabel=r'$q^2 ADD, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist")

                plt_name = 'MC vs smeared MC q2 ADD'
                hists = {"data hist": data_df['q2_ADD'], "mc hist": jpsi_df["q2_ADD_smeared"]}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 10),
                           save_file='q2_ADD',
                           xlabel=r'$q^2 ADD, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist")

                # =================================================================================

                data_e_plus_nonzero_brem_E = data_df.query('e_plus_BremMultiplicity > 0.')['e_plus_BREM_P']
                MC_e_plus_nonzero_brem_E = jpsi_df.query('e_plus_BremMultiplicity > 0.')['e_plus_BREM_P']

                plt_name = 'data vs MC non-zero brem E log'
                hists = {"data hist": data_e_plus_nonzero_brem_E / 1000., "mc hist": MC_e_plus_nonzero_brem_E / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0., 100.),
                           save_file='nonzero_bremE',
                           xlabel=r'$E_{\gamma}, GeV$',
                           ylabel='Normed events / 2 GeV',
                           title=plt_name,
                           stand_hist="data hist",
                           log=True)

                plt_name = 'data vs MC 1-cosTheta lin'
                hists = {"data hist": 1 - data_df['cosTheta'], "mc hist": 1 - jpsi_df['cosTheta']}
                plot_hists(hists, tags=tags, nbins=30, bin_range=(0.0001, 0.1),
                           save_file='1-cosTheta lin',
                           xlabel=r'$1 - cos(\theta)$',
                           ylabel='Normed events',
                           title=plt_name,
                           stand_hist=None,
                           log=False,
                           x_log=True)

                # =================================================================================

                plt_name = 'data vs smeared MC e_plus P'
                hists = {"data hist": data_df['e_plus_P'] / 1000.,
                         "smeared mc hist": jpsi_df["e_plus_P_smeared"] / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 300),
                           save_file='e_plus_P linear',
                           xlabel=r'$p_{e^+}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="data hist",
                           log=False)

                plt_name = 'data vs MC e_plus P'
                hists = {"data hist": data_df['e_plus_P'] / 1000., "mc hist": jpsi_df['e_plus_P'] / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 300),
                           save_file='e_plus_P linear',
                           xlabel=r'$p_{e^+}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="data hist",
                           log=False)

                plt_name = 'MC vs smeared MC e_plus P'
                hists = {"mc hist": jpsi_df['e_plus_P'] / 1000., "smeared mc hist": jpsi_df["e_plus_P_smeared"] / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 300),
                           save_file='e_plus_P linear',
                           xlabel=r'$p_{e^+}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="mc hist",
                           log=False)

                # =================================================================================

                plt_name = 'data vs smeared MC e_plus TRACK_P'
                hists = {"data hist": data_df['e_plus_TRACK_P'] / 1000.,
                         "smeared mc hist": jpsi_df["e_plus_TRACK_P_smeared"] / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 300),
                           save_file='e_plus_TRACK_P linear',
                           xlabel=r'$p_{e^+}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="data hist",
                           log=False)

                plt_name = 'data vs MC e_plus TRACK_P'
                hists = {"data hist": data_df['e_plus_TRACK_P'] / 1000., "mc hist": jpsi_df['e_plus_TRACK_P'] / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 300),
                           save_file='e_plus_TRACK_P linear',
                           xlabel=r'$p_{e^+}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="data hist",
                           log=False)

                plt_name = 'MC vs smeared MC e_plus TRACK_P'
                hists = {"mc hist": jpsi_df['e_plus_TRACK_P'] / 1000.,
                         "smeared mc hist": jpsi_df["e_plus_TRACK_P_smeared"] / 1000.}
                plot_hists(hists, tags=tags, nbins=55, bin_range=(-25, 300),
                           save_file='e_plus_TRACK_P linear',
                           xlabel=r'$p_{e^+}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="mc hist",
                           log=False)

                # =================================================================================

                plt_name = 'data vs MC ECAL region'
                hists = {"data hist": data_df['e_plus_L0Calo_ECAL_region'],
                         "mc hist": jpsi_df['e_plus_L0Calo_ECAL_region'] }
                plot_hists(hists, tags=tags, nbins=3, bin_range=(0, 3),
                           save_file='ECAL region',
                           xlabel=r'$e^+  ECAL  region$',
                           ylabel='Normed events / 1',
                           title=plt_name,
                           stand_hist="data hist",
                           log=False)

                # =================================================================================

                plt_name = 'data vs smeared MC e_minus P'
                hists = {"data hist": data_df['e_minus_P'] / 1000.,
                         "smeared mc hist": jpsi_df["e_minus_P_smeared"] / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 300),
                           save_file='e_minus_P linear',
                           xlabel=r'$p_{e^-}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="data hist",
                           log=False)

                plt_name = 'data vs MC e_minus P'
                hists = {"data hist": data_df['e_minus_P'] / 1000., "mc hist": jpsi_df['e_minus_P'] / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 300),
                           save_file='e_minus_P linear',
                           xlabel=r'$p_{e^-}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="data hist",
                           log=False)

                plt_name = 'MC vs smeared MC e_minus P'
                hists = {"mc hist": jpsi_df['e_minus_P'] / 1000.,
                         "smeared mc hist": jpsi_df["e_minus_P_smeared"] / 1000.}
                plot_hists(hists, tags=tags, nbins=50, bin_range=(0, 300),
                           save_file='e_minus_P linear',
                           xlabel=r'$p_{e^-}, [GeV/c^2]$',
                           ylabel='Normed events / 6',
                           title=plt_name,
                           stand_hist="mc hist",
                           log=False)

                plt_name = 'MC brem v track momentum correlation'

                plot_2dhist(jpsi_df.query('e_plus_BremMultiplicity > 0.')['e_plus_TRACK_P_smeared'] / 1000.,
                            jpsi_df.query('e_plus_BremMultiplicity > 0.')['e_plus_BREM_P'] / 1000.,
                            bins=200, range=((0, 100), (0, 100)), xlabel='track P smeared', yalbel='brem P', clabel='Events',
                            filename=f'{brem_tag}_{trig_tag}_Brem_v_track_sm_MC_momentum')

                plot_2dhist(jpsi_df.query('e_plus_BremMultiplicity > 0.')['q2_nobrem'], MC_e_plus_nonzero_brem_E/1000.,
                            bins=(50, 200), range=((0, 10), (0, 100)), xlabel='track q2', yalbel='brem P', clabel='Events',
                            filename=f'{brem_tag}_{trig_tag}_Brem_momentum_v_track_q2_MC')

                # ===========================================================================
                # e_plus MC

                print(jpsi_df.query('e_plus_BREM_P > 0.')['e_plus_BREM_P']/1000.)

                plot_2dhist(jpsi_df.query('e_plus_BremMultiplicity > 0.')['e_plus_TRACK_P']/1000.,
                            jpsi_df.query('e_plus_BremMultiplicity > 0.')['e_plus_BREM_P']/1000.,
                            bins=(40, 10), range=((0, 40), (0, 10)), xlabel='track P', yalbel='brem P', clabel='Events',
                            path='../Results/Hists2d/Correlations/MC/e_plus',
                            filename=f'{brem_tag}_{trig_tag}_Brem_v_track_MC_momentum')

                plot_2dhist(jpsi_df.query('e_plus_BremMultiplicity > 0.')['e_plus_BREM_P']/1000.,
                            jpsi_df.query('e_plus_BremMultiplicity > 0.')['cosTheta'],
                            bins=[10, 15], range=((0, 10), (0.85, 1.)), xlabel='brem P', yalbel='cosTheta', clabel='Events',
                            path='../Results/Hists2d/Correlations/MC/e_plus',
                            filename=f'{brem_tag}_{trig_tag}_BremP_v_cosTheta_MC')

                plot_2dhist(jpsi_df.query('e_plus_BremMultiplicity > 0.')['e_plus_TRACK_P']/1000.,
                            jpsi_df.query('e_plus_BremMultiplicity > 0.')['cosTheta'],
                            bins=[40, 15], range=((0, 40), (0.85, 1.)), xlabel='track P', yalbel='cosTheta', clabel='Events',
                            path='../Results/Hists2d/Correlations/MC/e_plus',
                            filename=f'{brem_tag}_{trig_tag}_TrackP_v_cosTheta_MC_brem_emitted')

                plot_2dhist(jpsi_df['e_plus_TRACK_P']/1000.,
                            jpsi_df['cosTheta'],
                            bins=[40, 15], range=((0, 40), (0.85, 1.)), xlabel='track P', yalbel='cosTheta', clabel='Events',
                            path='../Results/Hists2d/Correlations/MC/e_plus',
                            filename=f'{brem_tag}_{trig_tag}_TrackP_v_cosTheta_MC')

                # ===========================================================================
                # e_plus Data

                plot_2dhist(data_df.query('e_plus_BremMultiplicity > 0.')['e_plus_TRACK_P'] / 1000.,
                            data_df.query('e_plus_BremMultiplicity > 0.')['e_plus_BREM_P'] / 1000.,
                            bins=(40, 10), range=((0, 40), (0, 10)), xlabel='track P', yalbel='brem P', clabel='Events',
                            path='../Results/Hists2d/Correlations/Data/e_plus',
                            filename=f'{brem_tag}_{trig_tag}_BremP_v_trackP_Data')

                plot_2dhist(data_df.query('e_plus_BremMultiplicity > 0.')['e_plus_BREM_P'] / 1000.,
                            data_df.query('e_plus_BremMultiplicity > 0.')['cosTheta'],
                            bins=[10, 15], range=((0, 10), (0.85, 1.)), xlabel='brem P', yalbel='cosTheta',
                            clabel='Events',
                            path='../Results/Hists2d/Correlations/Data/e_plus',
                            filename=f'{brem_tag}_{trig_tag}_BremP_v_cosTheta_Data')

                plot_2dhist(data_df.query('e_plus_BremMultiplicity > 0.')['e_plus_TRACK_P'] / 1000.,
                            data_df.query('e_plus_BremMultiplicity > 0.')['cosTheta'],
                            bins=[40, 15], range=((0, 40), (0.85, 1.)), xlabel='track P', yalbel='cosTheta',
                            clabel='Events',
                            path='../Results/Hists2d/Correlations/Data/e_plus',
                            filename=f'{brem_tag}_{trig_tag}_TrackP_v_cosTheta_Data_brem_emitted')

                plot_2dhist(data_df['e_plus_TRACK_P'] / 1000.,
                            data_df['cosTheta'],
                            bins=[40, 15], range=((0, 40), (0.85, 1.)), xlabel='track P', yalbel='cosTheta',
                            clabel='Events',
                            path='../Results/Hists2d/Correlations/Data/e_plus',
                            filename=f'{brem_tag}_{trig_tag}_TrackP_v_cosTheta_Data')

                # ===============================================================================
                # ===========================================================================
                # e_minus MC

                plot_2dhist(jpsi_df.query('e_minus_BremMultiplicity > 0.')['e_minus_TRACK_P']/1000.,
                            jpsi_df.query('e_minus_BremMultiplicity > 0.')['e_minus_BREM_P']/1000.,
                            bins=(40, 10), range=((0, 40), (0, 10)), xlabel='track P', yalbel='brem P', clabel='Events',
                            path='../Results/Hists2d/Correlations/MC/e_minus',
                            filename=f'{brem_tag}_{trig_tag}_Brem_v_track_MC_momentum')

                plot_2dhist(jpsi_df.query('e_minus_BremMultiplicity > 0.')['e_minus_BREM_P']/1000.,
                            jpsi_df.query('e_minus_BremMultiplicity > 0.')['cosTheta'],
                            bins=[10, 15], range=((0, 10), (0.85, 1.)), xlabel='brem P', yalbel='cosTheta', clabel='Events',
                            path='../Results/Hists2d/Correlations/MC/e_minus',
                            filename=f'{brem_tag}_{trig_tag}_BremP_v_cosTheta_MC')

                plot_2dhist(jpsi_df.query('e_minus_BremMultiplicity > 0.')['e_minus_TRACK_P']/1000.,
                            jpsi_df.query('e_minus_BremMultiplicity > 0.')['cosTheta'],
                            bins=[40, 15], range=((0, 40), (0.85, 1.)), xlabel='track P', yalbel='cosTheta', clabel='Events',
                            path='../Results/Hists2d/Correlations/MC/e_minus',
                            filename=f'{brem_tag}_{trig_tag}_TrackP_v_cosTheta_MC_brem_emitted')

                plot_2dhist(jpsi_df['e_minus_TRACK_P']/1000.,
                            jpsi_df['cosTheta'],
                            bins=[40, 15], range=((0, 40), (0.85, 1.)), xlabel='track P', yalbel='cosTheta', clabel='Events',
                            path='../Results/Hists2d/Correlations/MC/e_minus',
                            filename=f'{brem_tag}_{trig_tag}_TrackP_v_cosTheta_MC')

                # ===========================================================================
                # e_minus Data

                plot_2dhist(data_df.query('e_minus_BremMultiplicity > 0.')['e_minus_TRACK_P'] / 1000.,
                            data_df.query('e_minus_BremMultiplicity > 0.')['e_minus_BREM_P'] / 1000.,
                            bins=(40, 10), range=((0, 40), (0, 10)), xlabel='track P', yalbel='brem P', clabel='Events',
                            path='../Results/Hists2d/Correlations/Data/e_minus',
                            filename=f'{brem_tag}_{trig_tag}_BremP_v_trackP_Data')

                plot_2dhist(data_df.query('e_minus_BremMultiplicity > 0.')['e_minus_BREM_P'] / 1000.,
                            data_df.query('e_minus_BremMultiplicity > 0.')['cosTheta'],
                            bins=[10, 15], range=((0, 10), (0.85, 1.)), xlabel='brem P', yalbel='cosTheta',
                            clabel='Events',
                            path='../Results/Hists2d/Correlations/Data/e_minus',
                            filename=f'{brem_tag}_{trig_tag}_BremP_v_cosTheta_Data')

                plot_2dhist(data_df.query('e_minus_BremMultiplicity > 0.')['e_minus_TRACK_P'] / 1000.,
                            data_df.query('e_minus_BremMultiplicity > 0.')['cosTheta'],
                            bins=[40, 15], range=((0, 40), (0.85, 1.)), xlabel='track P', yalbel='cosTheta',
                            clabel='Events',
                            path='../Results/Hists2d/Correlations/Data/e_minus',
                            filename=f'{brem_tag}_{trig_tag}_TrackP_v_cosTheta_Data_brem_emitted')

                plot_2dhist(data_df['e_minus_TRACK_P'] / 1000.,
                            data_df['cosTheta'],
                            bins=[40, 15], range=((0, 40), (0.85, 1.)), xlabel='track P', yalbel='cosTheta',
                            clabel='Events',
                            path='../Results/Hists2d/Correlations/Data/e_minus',
                            filename=f'{brem_tag}_{trig_tag}_TrackP_v_cosTheta_Data')

                data_df, jpsi_df = reweight(data_df, jpsi_df)

                plt_name = 'reweighted data vs MC 1-cosTheta lin'
                hists = {"data hist": 1 - data_df['cosTheta'], "mc hist": 1 - jpsi_df['cosTheta']}
                weights = {"data hist": data_df['Weights'], "mc hist": jpsi_df['Weights']}
                plot_hists(hists, tags=tags, nbins=31, bin_range=(0.0001, 0.2),
                           save_file='1-cosTheta lin',
                           xlabel=r'$1 - cos(\theta)$',
                           ylabel='Normed events',
                           title=plt_name,
                           stand_hist=None,
                           log=False,
                           x_log=True,
                           weights=weights)

                # ===========================================

                plt_name = 'reweighted data vs MC 1-cosTheta lin'
                hists = {"data hist": 1 - data_df['cosTheta'], "mc hist": 1 - jpsi_df['cosTheta']}
                weights = {"data hist": data_df['Weights'], "mc hist": jpsi_df['Weights']}
                plot_hists(hists, tags=tags, nbins=31, bin_range=(0.0001, 0.2),
                           save_file='1-cosTheta lin',
                           xlabel=r'$1 - cos(\theta)$',
                           ylabel='Normed events',
                           title=plt_name,
                           stand_hist=None,
                           log=False,
                           x_log=True,
                           weights=weights)

                # ======================================================

                plt_name = 'reweighted data vs smeared MC q2'
                weights = {"data hist": data_df['Weights'], "smeared mc hist": jpsi_df['Weights']}
                hists = {"data hist": data_df['q2'], "smeared mc hist": jpsi_df["q2_smeared"]}
                plot_hists(hists, tags=tags, nbins=35, bin_range=(6, 13),
                           save_file='q2 assume factor',
                           xlabel=r'$q^2, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist",
                           weights=weights)

                plt_name = 'reweighted data vs MC q2'
                weights = {"data hist": data_df['Weights'], "mc hist": jpsi_df['Weights']}
                hists = {"data hist": data_df['q2'], "mc hist": jpsi_df['q2']}
                plot_hists(hists, tags=tags, nbins=35, bin_range=(6, 13),
                           save_file='q2 assume factor',
                           xlabel=r'$q^2, [GeV^2/c^4]$',
                           ylabel='Normed events / 0.2',
                           title=plt_name,
                           stand_hist="data hist",
                           weights=weights)



def smear_TRACK_P(data):
    data['q2_nobrem_smeared'] = np.power(data['J_psi_1S_TRACK_M_smeared'], 2) / 10 ** 6
    data['smear_factor'] = np.sqrt(np.divide(data['q2_nobrem_smeared'], data['q2_nobrem']))
    # data['smear_factor'] = np.ones_like(data['q2_nobrem_smeared'])
    data['e_plus_TRACK_P_smeared'] = np.multiply(data['e_plus_TRACK_P'], data['smear_factor'])
    data['e_minus_TRACK_P_smeared'] = np.multiply(data['e_minus_TRACK_P'], data['smear_factor'])
    return data


def calc_smeared_q2(data):
    data['e_plus_P_smeared'] = data['e_plus_TRACK_P_smeared'] + data['e_plus_BREM_P']
    data['e_minus_P_smeared'] = data['e_minus_TRACK_P_smeared'] + data['e_minus_BREM_P']
    data['q2_smeared'] = 2 * np.multiply(np.multiply(data['e_plus_P_smeared'], data['e_minus_P_smeared']),
                                         np.ones_like(data['cosTheta']) - data['cosTheta']) / 10 ** 6
    data['q2_ADD_smeared'] = data['q2_smeared'] - data['q2_nobrem_smeared']
    data['q2_halfsmeared'] = data['q2_nobrem_smeared'] + data['q2_ADD']
    return data


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


smear_q2_TRACK(data)
