from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import uproot
import csv
import time
from matplotlib.ticker import AutoMinorLocator
from pandas import Series, DataFrame
from rkhq import cuts, parse_name
import os

from Hist_Settings import hist_dict, data_branches, mc_branches

# General functions for the first step of resolution analysis. Extracting ntuples, creating new variables, cutting.
# The reading hierarchy: file -> sample -> year -> run. The final product is a dictionary with run numbers as keys
# that contains 3 data frames: data, rare_MC and Jpsi_MC (all the years within a run are united)
fraction = 1.

samples = {"data": "B2Kee_",
           "rare_MC": "B2Kee_",
           "Jpsi_MC": "B2KJpsi_"}

samples_keys = ["data", "rare_MC", "Jpsi_MC"]

common_path = "/media/roman/Backup Plus/resolution_data/"  # !!! change this to your data directory path !!!


def extract_from_vector(x):
    try:
        return x[0]
    except TypeError:
        return x


def to_GeVsq(x):
    return x / 1000000.


def calc_q2(x):
    return x ** 2


def add_vars(x, y):
    return x + y


def calc_true_q2(dilep_PE, dilep_PX, dilep_PY, dilep_PZ):
    return dilep_PE ** 2 - (dilep_PX ** 2 + dilep_PY ** 2 + dilep_PZ ** 2)


def calc_q2_res(q2, true_q2):
    return (q2 - true_q2) / true_q2


def calc_cosTheta(e_plus_px, e_plus_py, e_plus_pz, e_plus_p, e_minus_px, e_minus_py, e_minus_pz, e_minus_p):
    dot_product = e_plus_px * e_minus_px + e_plus_py * e_minus_py + e_plus_pz * e_minus_pz
    cosTheta = dot_product / (e_plus_p * e_minus_p)
    return cosTheta


def read_file(path, sample, branches):
    print("=====")
    print("Processing {0} file".format(sample))
    with uproot.open(path) as file:
        tree = file["DecayTree"]
        df = tree.arrays(branches, library='pd')

    num_before_cuts = len(df.index)
    print("Events before cuts: {0}".format(num_before_cuts))

    df["B_plus_DTFM_M"] = df["B_plus_DTFM_M"].apply(extract_from_vector)
    df["q2"] = df["J_psi_1S_M"].apply(calc_q2)
    df["q2"] = df["q2"].apply(to_GeVsq)
    df['e_plus_BREM_P'] = df['e_plus_P'] - df['e_plus_TRACK_P']
    df['e_minus_BREM_P'] = df['e_minus_P'] - df['e_minus_TRACK_P']
    df['cosTheta'] = np.vectorize(calc_cosTheta)(df.e_plus_PX, df.e_plus_PY, df.e_plus_PZ, df.e_plus_P,
                                                 df.e_minus_PX, df.e_minus_PY, df.e_minus_PZ, df.e_minus_P)
    df = df.query("q2 > 6.0 and q2 < 12.96")  # according to LHCb-ANA-2017-042, Section 5.4.1, table 3
    df = df.query("B_plus_DTFM_M > 5200 and B_plus_DTFM_M < 5680")  # same, Section 6.9 (stricter cut)
    df = df.query("BDT_score_selection >= 0.8")  # 85% of signal

    df.drop(['e_plus_PX', 'e_plus_PY', 'e_plus_PZ', 'e_minus_PX', 'e_minus_PY', 'e_minus_PZ'],
            axis=1,
            inplace=True)

    num_after_cuts = len(df.index)
    print("Number of events after cuts: {0}".format(num_after_cuts))
    return df


def read_rare_MC(path, sample, branches):
    print("=====")
    print("Processing {0} file".format(sample))
    with uproot.open(path) as file:
        tree = file["DecayTree"]
        df = tree.arrays(branches, library='pd')

    num_before_cuts = len(df.index)
    print("Events before cuts: {0}".format(num_before_cuts))

    df["B_plus_DTFM_M"] = df["B_plus_DTFM_M"].apply(extract_from_vector)
    df["q2"] = df["J_psi_1S_M"].apply(calc_q2)
    df["q2"] = df["q2"].apply(to_GeVsq)
    df["q2_nobrem"] = (df['J_psi_1S_TRACK_M'] / 1000) ** 2

    df['e_plus_BREM_P'] = df['e_plus_P'] - df['e_plus_TRACK_P']
    df['e_minus_BREM_P'] = df['e_minus_P'] - df['e_minus_TRACK_P']
    df['cosTheta'] = np.vectorize(calc_cosTheta)(df.e_plus_PX, df.e_plus_PY, df.e_plus_PZ, df.e_plus_P,
                                                 df.e_minus_PX, df.e_minus_PY, df.e_minus_PZ, df.e_minus_P)

    df = df.query("B_plus_M > 4880 and B_plus_M < 6200")  # same, Section 6.9 (stricter cut)
    df = df.query("BDT_score_selection >= 0.8")  # 85% of signal

    df.drop(['e_plus_PX', 'e_plus_PY', 'e_plus_PZ', 'e_minus_PX', 'e_minus_PY', 'e_minus_PZ'],
            axis=1,
            inplace=True)

    num_after_cuts = len(df.index)
    print("Number of events after cuts: {0}".format(num_after_cuts))
    return df


def create_new_vars(df, sample):
    print("=====")
    print("Creating new variables")

    # add brem/no_brem variables here if needed

    if "MC" in sample:
        df["q2_TRUE"] = df["J_psi_1S_M_TRUE"].apply(calc_q2)
        df["q2_TRUE"] = df["q2_TRUE"].apply(to_GeVsq)

    elif "data" in sample:
        df["J_psi_1S_M_TRUE"] = 3096.9  # pdg
        df["B_plus_M_TRUE"] = 5279.26  # pdg
        df["q2_TRUE"] = (3096.9 / 1000) ** 2

    df["q2_nobrem"] = (df['J_psi_1S_TRACK_M'] / 1000) ** 2

    # let's get the fits first
    return df


def read_sample(sample, year):  # right now reads 1 sample from 2 ntuples (idk how to merge them together)
    print("###==========###")
    print("Processing: {0} sample".format(sample))

    start = time.time()
    frames = []

    if sample == "data":
        prefix = "Data/data_"
        strip = ""
        truth = ""

    else:
        prefix = "MC/mc_"
        strip = ""
        truth = "truth_"

    # todo maybe put this into infofile?
    fold = "folded_"
    trig = "fullTrig_"
    presel = "fullPresel_"
    bdt = "bdt"

    mag_pols = ["MD_", "MU_"]

    for mag_pol in mag_pols:  # if MU and MD merged change path and delete for-loop, temp_df, frames, mag_pols and
        # concat

        # using the following naming scheme:

        path = common_path + prefix + samples[sample] + year + "_" + mag_pol + strip + fold \
               + truth + trig + presel + bdt + ".root"

        # example data file:
        # Path/To/Ntuples/Data/data_B2Kee_2018_MU_Strip34_folded_fullTrig_fullPresel_bdt.root
        # example MC file:
        # Path/To/Ntuples/MC/mc_B2KJpsi_2016_MD_folded_truth_fullTrig_fullPresel_bdt.root

        if not os.path.exists(path):
            print("Didn't find {0} file dated year {1}".format(sample, year))
            break

        if sample == "data":
            branches = data_branches
        elif "MC" in sample:
            branches = mc_branches

        if 'rare' not in sample:
            temp_df = read_file(path, sample, branches)
        else:
            temp_df = read_rare_MC(path, sample, branches)
        temp_df = create_new_vars(temp_df, sample)
        frames.append(temp_df)

    if os.path.exists(path):
        print("###==========###")
        print("Finished processing {0} samples".format(sample))
        print("Time elapsed: {0} seconds".format(time.time() - start))
        df_sample = pandas.concat(frames)
        return df_sample

    else:
        return None


def get_data_given_year(year):
    print("=====")
    print("Processing data from year {0}".format(year))
    data_year = {}
    for s in samples_keys:
        data_year[s] = read_sample(s, year)
    return data_year


def get_data_given_run(run):
    print("=====")
    print("Processing data from run {0}".format(run))
    temp_data_run = {}
    data_run = {}
    if run == 1:
        years = ["2011", "2012"]
    elif run == 2:
        years = ["2015", "2016", "2017", "2018"]
    elif run not in (1, 2):
        raise ValueError(f"Error! Run number {run} not in (1, 2).")
    for year in years:
        temp_data_run[year] = get_data_given_year(year)

    # need to concat all the same sample dfs within a run into one
    for s in samples_keys:
        same_sample_dfs = []
        for y in years:
            if temp_data_run[y][s] is not None:
                same_sample_dfs.append(temp_data_run[y][s])
        if same_sample_dfs:
            df_run = pandas.concat(same_sample_dfs)
            data_run[s] = df_run
    return data_run


def get_data_from_files():
    # switch_run = int(input("Please choose a data taking period to analyse.\n0 for all, 1 for run 1, 2 for run 2.\n"))
    data = {}
    switch_run = 0
    if switch_run == 0:
        for run in (1, 2):
            data["run" + str(run)] = get_data_given_run(run)
    elif switch_run == 1 or switch_run == 2:
        data["run" + str(switch_run)] = get_data_given_run(switch_run)
    else:
        raise ValueError("Error! Incorrect choice. Needed a number in (0, 1, 2)")
    problematic_runs = []
    for run, run_df in data.items():
        if not run_df:
            print(f"The entire set of ntuples of {run} is missing. Proceeding with another run if possible")
            problematic_runs.append(run)
    for p_run in problematic_runs:
        del data[p_run]
    if data:
        return data
    else:
        raise FileNotFoundError("All the data for the chosen run or runs is missing")


# each sample must be broken into different brem and trigger categories. 2 trigger cats, 3 brem cats
# The structure of of data is: data[run_number][sample]

def categorize_by_brem(data):
    print("=====")
    print("Dividing data into brem categories")

    brem_zero_cut = "e_plus_BremMultiplicity == 0 and e_minus_BremMultiplicity == 0"

    brem_one_cut = "(e_plus_BremMultiplicity == 0 and e_minus_BremMultiplicity == 1) or " \
                   "(e_plus_BremMultiplicity == 1 and e_minus_BremMultiplicity == 0)"

    brem_two_cut = "(e_plus_BremMultiplicity == 0 and e_minus_BremMultiplicity > 1) or " \
                   "(e_plus_BremMultiplicity > 1 and e_minus_BremMultiplicity == 0) or " \
                   "(e_plus_BremMultiplicity >= 1 and e_minus_BremMultiplicity >= 1)"

    for run_num, data_run in data.items():
        print("Processing run {0} data".format(run_num))
        for s, data_sample in data_run.items():
            print("Processing {0} sample".format(s))

            brem0_df = data_sample.copy()
            brem0_df = brem0_df.query(brem_zero_cut)
            brem0_df["brem_cat"] = "brem_zero"

            brem1_df = data_sample.copy()
            brem1_df = brem1_df.query(brem_one_cut)
            brem1_df["brem_cat"] = "brem_one"

            brem2_df = data_sample.copy()
            brem2_df = brem2_df.query(brem_two_cut)
            brem2_df["brem_cat"] = "brem_two"

            b_zero_n = len(brem0_df.index)
            b_one_n = len(brem1_df.index)
            b_two_n = len(brem2_df.index)

            print("Events in different categories:\nBrem zero: {0}"
                  "\nBrem one:  {1}\nBrem two:  {2}".format(b_zero_n, b_one_n, b_two_n))

            sample_df = pandas.concat([brem0_df, brem1_df, brem2_df])
            data[run_num][s] = sample_df.copy()
            del sample_df
    return data


def categorize_by_trig(data):
    print("=====")
    print("Dividing data into trig categories")

    for run_num, data_run in data.items():
        print(f"Processing run {run_num} data")
        for s, data_sample in data_run.items():
            print(f"Processing {s} sample")

            TIS_cut = cuts.get_cuts(name="L0TISnoeTS", year=run_num, channel="electron")
            eTOS_cut = cuts.get_cuts(name='L0eTOSnoTIS', year=run_num, channel="electron")

            TIS_cut = TIS_cut.to_numexpr()
            eTOS_cut = eTOS_cut.to_numexpr()

            TIS_df = data_sample.copy()
            TIS_df = TIS_df.query(TIS_cut)
            TIS_df["trig_cat"] = "TIS"

            eTOS_df = data_sample.copy()
            eTOS_df = eTOS_df.query(eTOS_cut)
            eTOS_df["trig_cat"] = "eTOS"

            eTOS_n = len(eTOS_df.index)
            TIS_n = len(TIS_df.index)

            print("Events in different categories:\neTOS: {0}\nTIS:  {1}".format(eTOS_n, TIS_n))
            sample_df = pandas.concat([TIS_df, eTOS_df])
            data[run_num][s] = sample_df.copy()
            del sample_df
    return data


# Now let's plot histograms

def plot_histogram(data, tags, plt_name):
    run_num = tags["run_num"]
    s = tags["sample"]
    if "brem_cat" in tags:
        b = tags["brem_cat"]
    else:
        b = "all brem cats"
    if "trig_cat" in tags:
        t = tags["trig_cat"]
    else:
        t = "all trig cats"

    plot_label = hist_dict[plt_name]["plot_label"] + "\nrun_" + run_num + '\n' + b + "\n" + t
    h_bin_width = hist_dict[plt_name]["bin_width"]
    h_num_bins = hist_dict[plt_name]["num_bins"]
    h_xmin = hist_dict[plt_name]["x_min"]
    h_xmax = hist_dict[plt_name]["x_max"]
    h_xlabel = hist_dict[plt_name]["x_label"]
    h_ylabel = hist_dict[plt_name]["y_label"]
    x_var = hist_dict[plt_name]["x_var"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    data_x, _ = np.histogram(data[x_var].values, bins=bins)

    plt.clf()
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    main_axes.errorbar(bin_centers, data_x, xerr=h_bin_width / 2, fmt="ok", label=s)

    main_axes.legend(title=plot_label, loc="best")
    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel(h_ylabel)
    main_axes.set_xlabel(h_xlabel)
    # plt.show()
    plt.savefig("../Output/Hist_{0}_{1}_{2}_{3}.pdf".format(run_num, s, b, t))
