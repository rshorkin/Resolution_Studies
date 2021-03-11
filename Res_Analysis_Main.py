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
from Smearing import smear

# get data, tag it with brem and trigger tags
data = get_data_from_files()  # the result is {1: {"data": df1, "Jpsi_MC: df2, "rare_MC": df3}, 2: {...}}
data = categorize_by_brem(data)  # creates an additional column with brem tags
data = categorize_by_trig(data)  # creates an additional column with trig tags
zfit.run.set_graph_mode(False)


# create a few helper functions
def full_analysis_w_brem(_data):
    obs_dict = {"mee": zfit.Space('J_psi_1S_M', limits=(2520, 3600)),
                "mKee": zfit.Space('B_plus_M', limits=(4600, 5800))}

    data_select = int(input("Choose the type of analysis\n0 for all trigger and brem categories\n"
                            "1 for all brem categories (trigger independent)\n"
                            "2 for all trigger categories (brem independent) (NOT RECOMMENDED)\n"))  # not implemented

    if data_select == 0:
        trigger_tags = ["TIS", "eTOS"]
        brem_tags = ["brem_zero", "brem_one", "brem_two"]
        query_str = "brem_cat == @brem_tag and trig_cat == @trig_tag"
    elif data_select == 1:
        trigger_tags = ["TIS_&_eTOS"]
        brem_tags = ["brem_zero", "brem_one", "brem_two"]
        query_str = "brem_cat == @brem_tag"
    elif data_select == 2:
        trigger_tags = ["TIS", "eTOS"]
        brem_tags = ["brem_independent"]
        query_str = "trig_cat == @trig_tag"
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

    for run_tag, data_run in _data.items():
        jpsi_sample = data_run["Jpsi_MC"]
        data_sample = data_run["data"]

        models_by_brem = {"data": {}, "Jpsi_MC": {}, "smeared_MC": {}}
        smeared_MC_by_brem = []

        for brem_tag in brem_tags:
            initial_params = initial_params_dict[option][brem_tag]
            for trig_tag in trigger_tags:
                print("Working on {0} {1} samples".format(brem_tag, trig_tag))
                models = {}
                tags = {"run_num": str(run_tag), "brem_cat": brem_tag, "trig_cat": trig_tag}

                jpsi_df = jpsi_sample.query(query_str)
                print("MC EVENTS:", len(jpsi_df.index))
                tags["sample"] = "Jpsi_MC"
                # plot_histogram(jpsi_df, tags, tags["sample"] + "_" + option)    # test

                print("####==========####\nFitting MC")
                ini_model = create_initial_model(initial_params, obs, tags)
                models["original MC fit"] = ini_model
                mc_fit_params = initial_fitter(jpsi_df[x_var], ini_model, obs)
                # plot_fit_result(models, jpsi_df[x_var], obs, tags, tags["sample"] + "_" + option)  # test

                data_df = data_sample.query(query_str)
                tags["sample"] = "data"
                # plot_histogram(data_df, tags, tags["sample"] + "_" + option)   # test

                print("####==========####\nFitting data")
                data_fit_params, fin_models = create_data_fit_model(data_df[x_var], mc_fit_params, obs, tags)
                tags["run_num"] = "run2_data"
                models["data fit"] = fin_models["combined"]
                plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option)  # test
                # plot_fit_result(fin_models, data_df[x_var], obs, tags, tags["sample"] + "_" + option)  # test
                tags["run_num"] = "run2"

                print("####==========####\nSmearing MC")
                smearing_params = {"mass": mass}
                for param_name, param_value in mc_fit_params.items():
                    if "mu" in param_name:
                        smearing_params["mu"] = param_value
                for param_name, param_value in data_fit_params.items():
                    if "delta_mu" in param_name:
                        smearing_params["delta_mu"] = param_value["value"]
                    if "scale_sigma" in param_name:
                        smearing_params["scale_sigma"] = param_value["value"]

                jpsi_df = smear(jpsi_df, smearing_params, x_var)

                print("####==========####\nFitting smeared MC")
                tags["run_num"] = str(run_tag) + "_smeared"
                # if brem_tag != "brem_zero":
                # sm_model = create_initial_model(initial_params, obs, tags, switch="smearedMC",
                #     params=data_fit_params)
                # else:
                sm_model = create_initial_model(initial_params, obs, tags)
                models["smeared MC fit"] = sm_model
                _ = initial_fitter(jpsi_df[x_var + "_smeared"], sm_model, obs)

                plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option, data_fit_params)

                if data_select == 1:
                    models_by_brem["data"][brem_tag] = fin_models["combined"]
                    models_by_brem["Jpsi_MC"][brem_tag] = models["original MC fit"]
                    models_by_brem["smeared_MC"][brem_tag] = models["smeared MC fit"]
                    smeared_MC_by_brem.append(jpsi_df)

                hists = {"data hist": data_df[x_var], "smeared mc hist": jpsi_df[x_var + "_smeared"]}
                plot_hists(hists, tags["sample"] + "_" + option, tags)

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

            plot_fit_result(full_models, None, obs, tags, "data" + "_" + option)


def full_analysis_no_brem(_data):
    obs = zfit.Space('J_psi_1S_TRACK_M', limits=(300, 3100))
    models = {}
    option = "mee_nobrem"
    initial_params = initial_params_dict[option]
    x_var = "J_psi_1S_TRACK_M"
    for run_tag, data_run in _data.items():
        tags = {"run_num": str(run_tag), "brem_cat": "nobrem", "trig_cat": "all"}

        data_sample = data_run["data"]
        tags["sample"] = "data"
        plot_histogram(data_sample, tags, tags["sample"] + "_" + option)

        jpsi_sample = data_run["Jpsi_MC"]
        tags["sample"] = "Jpsi_MC"
        plot_histogram(jpsi_sample, tags, tags["sample"] + "_" + option)

        ini_model = create_initial_model(initial_params, obs, tags, switch="nobrem")
        models["initial MC fit"] = ini_model
        mc_fit_params = initial_fitter(jpsi_sample[x_var], ini_model, obs)
        plot_fit_result(models, jpsi_sample[x_var], obs, tags, tags["sample"] + "_" + option)


def plot_hists(dict, plt_name, tags):
    plt.figure()
    h_num_bins = hist_dict[plt_name]["num_bins"]
    h_xmin = hist_dict[plt_name]["x_min"]
    h_xmax = hist_dict[plt_name]["x_max"]
    h_xlabel = hist_dict[plt_name]["x_label"]
    h_ylabel = hist_dict[plt_name]["y_label"]
    for key, value in dict.items():
        plt.hist(value, bins=h_num_bins, alpha=0.5, range=(h_xmin, h_xmax), density=True, label=key)

    plt.legend(loc='upper right')
    plt.title("data vs smeared mc hist")
    plt.savefig(f"../Output/data_vs_smeared_mc_{tags['brem_cat']}_{tags['trig_cat']}.pdf")
    plt.close()

full_analysis_w_brem(data)
# full_analysis_no_brem(data)
