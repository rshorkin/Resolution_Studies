import tensorflow as tf
import zfit
import matplotlib.pyplot as plt
import numpy as np
import pandas

from Service import *
from Fitting import *

obs = zfit.Space('J_psi_1S_M', limits=(2200, 3800))

# get data, tag it with brem and trigger tags
data = get_data_from_files()  # the result is {1: {"data": df1, "Jpsi_MC: df2, "rare_MC": df3}, 2: {...}}
data = categorize_by_brem(data)  # creates an additional column with brem tags
data = categorize_by_trig(data)  # creates an additional column with trig tags


# create a few helper functions
def full_analysis(_data):
    trigger_tags = ["TIS", "eTOS"]
    brem_tags = ["brem_zero", "brem_one", "brem_two"]
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
                plot_histogram(jpsi_df, tags, "Jpsi_MC_mee")

                ini_model = create_initial_model(tags)
                models["ini_model"] = ini_model
                parameters = initial_fitter(jpsi_df["J_psi_1S_M"], ini_model)
                plot_fit_result(models, jpsi_df["J_psi_1S_M"], tags, "Jpsi_MC_mee")

                data_df = data_sample.query("brem_cat == @brem_tag and trig_cat == @trig_tag")
                tags["sample"] = "data"
                plot_histogram(data_df, tags, "data_mee")

                _, fin_model = create_data_fit_model(data_df["J_psi_1S_M"], parameters, tags)
                models["fin model"] = fin_model
                plot_fit_result(models, data_df["J_psi_1S_M"], tags, "data_mee")


full_analysis(data)
