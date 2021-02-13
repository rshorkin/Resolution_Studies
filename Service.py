import matplotlib.pyplot as plt
import numpy as np
import pandas
import math
import uproot
import csv
import time
from matplotlib.ticker import AutoMinorLocator

from Hist_Settings import hist_dict

# General functions for the first step of resolution analysis. Extracting ntuples, creating new variables, cutting.
# The reading hierarchy: file -> sample -> year -> run. The final product is a dictionary with run numbers as keys
# that contains 3 data frames: data, rare_MC and Jpsi_MC (all the years within a run are united)
fraction = 1.
samples = {"data": "B2Kee_",
           "rare_MC": "B2Kee_",
           "Jpsi_MC": "B2KJpsi_"}
samples_keys = ["data", "rare_MC", "Jpsi_MC"]
common_path = "/media/roman/Backup Plus/resolution_data/"  # !!! change this to your data directory path !!!
data_branches = ["J_psi_1S_M", "B_plus_DTFM_M", "BDT_score_selection", "e_plus_BremMultiplicity",
                 "e_minus_BremMultiplicity", "L0TISOnly_d", "L0ETOSOnly_d", "J_psi_1S_TRACK_M"]
mc_branches = ["J_psi_1S_M", "B_plus_DTFM_M", "BDT_score_selection", "J_psi_1S_M_TRUE", "e_plus_BremMultiplicity",
               "e_minus_BremMultiplicity", "L0TISOnly_d", "L0ETOSOnly_d", "J_psi_1S_TRACK_M"]


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


def read_file(path, sample, branches):
    print("=====")
    print("Processing {0} file".format(sample))
    mc = uproot.open(path)["DecayTree"]
    numevents = uproot.numentries(path, "DecayTree")
    df = mc.pandas.df(branches, flatten=False, entrystop=numevents * fraction)

    num_before_cuts = len(df.index)
    print("Events before cuts: {0}".format(num_before_cuts))

    df["B_plus_DTFM_M"] = df["B_plus_DTFM_M"].apply(extract_from_vector)
    df["q2"] = df["J_psi_1S_M"].apply(calc_q2)
    df["q2"] = df["q2"].apply(to_GeVsq)

    df = df.query("q2 > 6.0 and q2 < 12.96")  # according to LHCb-ANA-2017-042, Section 5.4.1, table 3
    df = df.query("B_plus_DTFM_M > 5200 and B_plus_DTFM_M < 5680")  # same, Section 6.9 (stricter cut)
    df = df.query("BDT_score_selection >= 0.85")  # should ask about this value

    num_after_cuts = len(df.index)
    print("Number of events after cuts: {0}".format(num_after_cuts))
    return df


def create_new_vars(df, sample):
    print("=====")
    print("Creating new variables")

    # add brem/no_brem variables here if needed

    if "MC" in sample:
        df["TRUE_q2"] = df["J_psi_1S_M_TRUE"].apply(calc_q2)
        df["TRUE_q2"] = df["TRUE_q2"].apply(to_GeVsq)

    elif "data" in sample:
        df["TRUE_q2"] = (3096.9 / 1000) ** 2

    # df["q2_res"] = np.vectorize(calc_q2_res)(q2, TRUE_q2)
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

        if sample == "data":
            branches = data_branches
        elif "MC" in sample:
            branches = mc_branches
        else:
            raise ValueError("Error! Didn't find {0} sample from {1} at {2}!".format(sample, year, path))

        temp_df = read_file(path, sample, branches)
        temp_df = create_new_vars(temp_df, sample)
        frames.append(temp_df)
    df_sample = pandas.concat(frames)

    print("###==========###")
    print("Finished processing {0} samples".format(sample))
    print("Time elapsed: {0} seconds".format(time.time() - start))
    return df_sample


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
        years = []  # must fill in once I know what years go where
    elif run == 2:
        years = ["2018"]  # same
    elif run not in (1, 2):
        raise ValueError("Error! Run number {0} not in (1, 2).".format(run))
    for year in years:
        temp_data_run[year] = get_data_given_year(year)

    # need to concat all the same sample dfs within a run into one
    for s in samples_keys:
        same_sample_dfs = []
        for y in years:
            same_sample_dfs.append(temp_data_run[y][s])
        df_run = pandas.concat(same_sample_dfs)
        data_run[s] = df_run
    return data_run


def get_data_from_files():
    switch_run = int(input("Please choose a data taking period to analyse.\n0 for all, 1 for run 1, 2 for run 2.\n"))
    data = {}
    if switch_run == 0:
        for run in (1, 2):
            data[run] = get_data_given_run(run)
    elif switch_run == 1 or switch_run == 2:
        data[switch_run] = get_data_given_run(switch_run)
    else:
        raise ValueError("Error! Incorrect choice. Needed a number in (0, 1, 2)")
    return data


# each sample must be broken into different brem and trigger categories. 2 trigger cats, 3 brem cats


def categorize_by_brem(data):
    print("=====")
    print("Dividing data into brem categories")

    brem_zero_cut = "e_plus_BremMultiplicity == 0 and e_minus_BremMultiplicity == 0"

    brem_one_cut = "(e_plus_BremMultiplicity == 0 and e_minus_BremMultiplicity == 1) or " \
                   "(e_plus_BremMultiplicity == 1 and e_minus_BremMultiplicity == 0)"

    brem_two_cut = "(e_plus_BremMultiplicity == 0 and e_minus_BremMultiplicity > 1) or " \
                   "(e_plus_BremMultiplicity > 1 and e_minus_BremMultiplicity == 0) or " \
                   "(e_plus_BremMultiplicity >= 1 and e_minus_BremMultiplicity >= 1)"
    temp_dict = {}
    for run_num, data_run in data.items():
        print("Processing run {0} data".format(run_num))
        for s, data_sample in data_run.items():
            print("Processing {0} sample".format(s))
            temp_dict["b_zero"] = data_sample.query(brem_zero_cut)
            temp_dict["b_one"] = data_sample.query(brem_one_cut)
            temp_dict["b_two"] = data_sample.query(brem_two_cut)

            b_zero_n = len(temp_dict["b_zero"].index)
            b_one_n = len(temp_dict["b_one"].index)
            b_two_n = len(temp_dict["b_two"].index)

            print("Events in different categories:\nBrem zero: {0}"
                  "\nBrem one:  {1}\nBrem two:  {2}".format(b_zero_n, b_one_n, b_two_n))

            data[run_num][s] = temp_dict.copy()
    return data


def categorize_by_trig(data):
    print("=====")
    print("Dividing data into trig categories")

    TIS_cut = "L0TISOnly_d == 1"
    eTOS_cut = "L0ETOSOnly_d == 1"

    temp_dict = {}
    for run_num, data_run in data.items():
        print("Processing run {0} data".format(run_num))
        for s, data_sample in data_run.items():
            print("Processing {0} sample".format(s))
            for b, data_brem in data_sample.items():
                print("Processing {0} brem category".format(b))
                temp_dict["TIS"] = data_brem.query(TIS_cut)
                temp_dict["eTOS"] = data_brem.query(eTOS_cut)

                eTOS_n = len(temp_dict["eTOS"].index)
                TIS_n = len(temp_dict["TIS"].index)

                print("Events in different categories:\neTOS: {0}\nTIS:  {1}".format(eTOS_n, TIS_n))

                data[run_num][s][b] = temp_dict.copy()
    return data


# I am not sure... there must be a better way to do it. Maybe some tags for each event and then query
# relevant combinations? Not sure whether it's cleaner...
# Anyway, the structure of of data is the following: data[run_number][sample][brem_cat][trig_cat]
# This way we could potentially concat trigger cats into one, then brem cats (if the need arises)

# Now let's plot histograms

def plot_histograms(data):
    print("###==========####")
    print("Started plotting histograms")

    for run_num, data_run in data.items():
        for s, data_sample in data_run.items():
            for b, data_brem in data_sample.items():
                for t, data_trig in data_brem.items():
                    plot_label = "$B \\rightarrow Kee$\n" + b + "\n" + t

                    h_bin_width = hist_dict["bin_width"]
                    h_num_bins = hist_dict["num_bins"]
                    h_xmin = hist_dict["x_min"]
                    h_xmax = hist_dict["x_max"]
                    h_xlabel = hist_dict["x_label"]
                    x_var = hist_dict["x_var"]

                    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
                    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

                    data_x, _ = np.histogram(data_trig[x_var].values, bins=bins)

                    plt.clf()
                    plt.axes([0.1, 0.30, 0.85, 0.65])
                    main_axes = plt.gca()
                    main_axes.errorbar(bin_centers, data_x, xerr=16, fmt="ok", label=s)

                    main_axes.legend(title=plot_label, loc="best")
                    main_axes.set_xlim(h_xmin, h_xmax)

                    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

                    main_axes.set_ylabel("Events/32 MeV")

                    main_axes.set_xlabel(h_xlabel)
                    # plt.show()
                    plt.savefig("../Output/Hist_run{0}_{1}_{2}_{3}.pdf".format(run_num, s, b, t))


# now for J/Psi MC fitting
