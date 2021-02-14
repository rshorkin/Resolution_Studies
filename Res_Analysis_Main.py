import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
import pandas

from Service import *
from Fitting import *

initial_params_dict = {"mee": {'mu': 3097., 'sigma': 20., 'alphal': 0.15, 'nl': 50., 'alphar': 0.9, 'nr': 3.},
                       "mKee": {'mu': 5250., 'sigma': 40., 'alphal': 0.4, 'nl': 24., 'alphar': 0.9, 'nr': 3.}}

obs_dict = {"mee": zfit.Space('J_psi_1S_M', limits=(2200, 3800)),
            "mKee": zfit.Space('B_plus_M', limits=(4600, 6200))}

# get data, tag it with brem and trigger tags
data = get_data_from_files()  # the result is {1: {"data": df1, "Jpsi_MC: df2, "rare_MC": df3}, 2: {...}}
data = categorize_by_brem(data)  # creates an additional column with brem tags
data = categorize_by_trig(data)  # creates an additional column with trig tags


# create a few helper functions
def full_analysis(_data):
    trigger_tags = ["TIS", "eTOS"]
    brem_tags = ["brem_zero", "brem_one", "brem_two"]

    switch = int(input("Please select between m(ee) and m(Kee) analysis\n0 for m(ee), 1 for m(Kee)\n"))
    if switch == 0:
        option = "mee"
        x_var = "J_psi_1S_M"
    elif switch == 1:
        option = "mKee"
        x_var = "B_plus_M"
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

                jpsi_df = jpsi_sample.query("brem_cat == @brem_tag and trig_cat == @trig_tag")
                tags["sample"] = "Jpsi_MC"
                plot_histogram(jpsi_df, tags, tags["sample"] + "_" + option)

                ini_model = create_initial_model(initial_params, obs, tags)
                models["ini_model"] = ini_model
                parameters = initial_fitter(jpsi_df[x_var], ini_model, obs)
                plot_fit_result(models, jpsi_df[x_var], obs, tags, tags["sample"] + "_" + option)

                data_df = data_sample.query("brem_cat == @brem_tag and trig_cat == @trig_tag")
                tags["sample"] = "data"
                plot_histogram(data_df, tags, tags["sample"] + "_" + option)

                _, fin_model = create_data_fit_model(data_df[x_var], parameters, obs, tags)
                models["fin model"] = fin_model
                plot_fit_result(models, data_df[x_var], obs, tags, tags["sample"] + "_" + option)


full_analysis(data)
