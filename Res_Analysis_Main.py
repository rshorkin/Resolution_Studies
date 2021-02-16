import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
import pandas

from Service import *
from Fitting import *
from Smearing import smear

initial_params_dict = {"mee": {'mu': 3097., 'sigma': 20., 'alphal': 0.15, 'nl': 50., 'alphar': 0.9, 'nr': 3.},
                       "mKee": {'mu': 5250., 'sigma': 40., 'alphal': 0.4, 'nl': 24., 'alphar': 0.9, 'nr': 3.}}

obs_dict = {"mee": zfit.Space('J_psi_1S_M', limits=(2200, 3800)),
            "mKee": zfit.Space('B_plus_M', limits=(4600, 6200))}

# get data, tag it with brem and trigger tags
data = get_data_from_files()  # the result is {1: {"data": df1, "Jpsi_MC: df2, "rare_MC": df3}, 2: {...}}
data = categorize_by_brem(data)  # creates an additional column with brem tags
data = categorize_by_trig(data)  # creates an additional column with trig tags
# zfit.run.set_graph_mode(False)


# create a few helper functions
def full_analysis(_data):
    data_select = int(input("Choose the type of analysis\n0 for all trigger and brem categories\n"
                            "1 for all brem categories (trigger independent)\n"
                            "2 for all trigger categories (brem independent) (NOT RECOMMENDED)\n"))

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
    initial_params = initial_params_dict[option]

    for run_tag, data_run in _data.items():
        jpsi_sample = data_run["Jpsi_MC"]
        data_sample = data_run["data"]
        for brem_tag in brem_tags:
            for trig_tag in trigger_tags:
                print("Working on {0} {1} samples".format(brem_tag, trig_tag))
                models = {}
                tags = {"run_num": str(run_tag), "brem_cat": brem_tag, "trig_cat": trig_tag}

                jpsi_df = jpsi_sample.query(query_str)
                print("MC EVENTS:", len(jpsi_df.index))
                tags["sample"] = "Jpsi_MC"
                plot_histogram(jpsi_df, tags, tags["sample"] + "_" + option)

                print("####==========####\nFitting MC")
                ini_model = create_initial_model(initial_params, obs, tags)
                models["initial model"] = ini_model
                mc_fit_params = initial_fitter(jpsi_df[x_var], ini_model, obs)
                plot_fit_result(models, jpsi_df[x_var], obs, tags, tags["sample"] + "_" + option)

                data_df = data_sample.query(query_str)
                tags["sample"] = "data"
                plot_histogram(data_df, tags, tags["sample"] + "_" + option)

                print("####==========####\nFitting data")
                data_fit_params, fin_models = create_data_fit_model(data_df[x_var], mc_fit_params, obs, tags)
                write_to_csv(data_fit_params, tags, tags["sample"] + "_" + option)
                models["final model"] = fin_models["combined"]
                plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option)
                tags["run_num"] = "2_test"
                plot_fit_result(fin_models, data_df[x_var], obs, tags, tags["sample"] + "_" + option)
                tags["run_num"] = "2"

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
                tags["run_num"] = str(run_tag) + "smeared"
                sm_model = create_initial_model(initial_params, obs, tags)
                models["smeared model"] = sm_model
                _ = initial_fitter(jpsi_df[x_var + "_smeared"], sm_model, obs)

                plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option)


full_analysis(data)
