initial_params_dict = {"mee": {
    "brem_zero": {'mu': 3100., 'sigma': 21., 'alphal': 0.2, 'nl': 3.5, 'alphar': 3.0, 'nr': 3.5},
    "brem_one": {'mu': 3100., 'sigma': 54., 'alphal': 0.23, 'nl': 3.5, 'alphar': 0.5, 'nr': 18.},
    "brem_two": {'mu': 3100., 'sigma': 42., 'alphal': 0.7, 'nl': 3.5, 'alphar': 0.3, 'nr': 3.5}},

    "mKee": {
        "brem_zero": {'mu': 5250., 'sigma': 30., 'alphal': 0.2, 'nl': 50., 'alphar': 3.0, 'nr': 3.5},
        "brem_one": {'mu': 5250., 'sigma': 40., 'alphal': 0.4, 'nl': 24., 'alphar': 0.9, 'nr': 3.},
        "brem_two": {'mu': 5300., 'sigma': 40., 'alphal': 0.4, 'nl': 24., 'alphar': 0.9, 'nr': 3.}}
}

data_branches = ["J_psi_1S_M", "B_plus_M", "B_plus_DTFM_M", "BDT_score_selection", "e_plus_BremMultiplicity",
                 "e_minus_BremMultiplicity", "L0TISOnly_d", "L0ETOSOnly_d", "J_psi_1S_TRACK_M",
                 'e_plus_L0Calo_ECAL_realET', 'e_minus_L0Calo_ECAL_realET', 'TCKCat', 'e_minus_L0ElectronDecision_TOS',
                 'B_plus_L0Global_TIS', 'e_plus_L0ElectronDecision_TOS', 'e_plus_L0Calo_ECAL_realET',
                 'e_minus_L0Calo_ECAL_realET', 'TCKCat', 'e_plus_L0ElectronDecision_TOS', 'B_plus_L0Global_TIS',
                 'e_minus_L0ElectronDecision_TOS']

mc_branches = ["J_psi_1S_M", "B_plus_M", "B_plus_DTFM_M", "BDT_score_selection", "J_psi_1S_M_TRUE",
               "e_plus_BremMultiplicity", "e_minus_BremMultiplicity", "L0TISOnly_d", "L0ETOSOnly_d",
               'e_plus_L0Calo_ECAL_realET', 'e_minus_L0Calo_ECAL_realET', 'TCKCat', 'e_minus_L0ElectronDecision_TOS',
               'B_plus_L0Global_TIS', 'e_plus_L0ElectronDecision_TOS', 'e_plus_L0Calo_ECAL_realET',
               'e_minus_L0Calo_ECAL_realET', 'TCKCat', 'e_plus_L0ElectronDecision_TOS', 'B_plus_L0Global_TIS',
               'e_minus_L0ElectronDecision_TOS', "J_psi_1S_TRACK_M", "B_plus_M_TRUE"]

hist_dict = {"Jpsi_MC_mee": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                             "bin_width": 32,
                             "num_bins": 40,
                             "x_min": 2520,
                             "x_max": 3800,
                             "x_label": "M(ee), MeV",
                             "y_label": "Events/32 MeV",
                             "x_var": "J_psi_1S_M"},
             "data_mee": {"plot_label": "$B \\rightarrow Kee$",
                          "bin_width": 32,
                          "num_bins": 40,
                          "x_min": 2520,
                          "x_max": 3800,
                          "x_label": "M(ee), MeV",
                          "y_label": "Events/32 MeV",
                          "x_var": "J_psi_1S_M"},
             "Jpsi_MC_mKee": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                              "bin_width": 32,
                              "num_bins": 50,
                              "x_min": 4600,
                              "x_max": 6200,
                              "x_label": "M(Kee), MeV",
                              "y_label": "Events/32 MeV",
                              "x_var": "B_plus_M"},
             "data_mKee": {"plot_label": "$B \\rightarrow Kee$",
                           "bin_width": 32,
                           "num_bins": 50,
                           "x_min": 4600,
                           "x_max": 6200,
                           "x_label": "M(Kee), MeV",
                           "y_label": "Events/32 MeV",
                           "x_var": "B_plus_M"},
             "Jpsi_MC_nobrem_mee": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                                    "bin_width": 70,
                                    "num_bins": 40,
                                    "x_min": 300,
                                    "x_max": 3100,
                                    "x_label": "M(ee) w/o brem recovery, MeV",
                                    "y_label": "Events/32 MeV",
                                    "x_var": "J_psi_1S_TRACK_M"},
             "data_nobrem_mee": {"plot_label": "$B \\rightarrow Kee$",
                                 "bin_width": 70,
                                 "num_bins": 40,
                                 "x_min": 300,
                                 "x_max": 3100,
                                 "x_label": "M(ee) w/o brem recovery, MeV",
                                 "y_label": "Events/32 MeV",
                                 "x_var": "J_psi_1S_TRACK_M"}
             }
