initial_params_dict = {
    "mee": {
        "brem_zero": {'mu': 3100., 'sigma': 21., 'alphal': 0.2, 'nl': 3.5, 'alphar': 3.0, 'nr': 3.5},
        "brem_one": {'mu': 3100., 'sigma': 54., 'alphal': 0.23, 'nl': 3.5, 'alphar': 0.5, 'nr': 18.},
        "brem_two": {'mu': 3100., 'sigma': 42., 'alphal': 0.7, 'nl': 3.5, 'alphar': 0.3, 'nr': 3.5}},

    "mKee": {
        "brem_zero": {'mu': 5250., 'sigma': 30., 'alphal': 0.2, 'nl': 50., 'alphar': 3.0, 'nr': 3.5},
        "brem_one": {'mu': 5250., 'sigma': 40., 'alphal': 0.4, 'nl': 24., 'alphar': 0.9, 'nr': 3.},
        "brem_two": {'mu': 5300., 'sigma': 40., 'alphal': 0.4, 'nl': 24., 'alphar': 0.9, 'nr': 3.}},

    "mee_nobrem": {
        "brem_zero": {'mu': 3100., 'sigma': 21., 'alphal': 0.2, 'nl': 3.5, 'alphar': 3.0, 'nr': 3.5},
        "brem_one": {'logn_mu': 6.2, 'logn_sigma': 0.57, 'logn_theta': -3180.,
                     'LCB_alpha': 2.14, 'LCB_n': 20., 'LCB_mu': 2200., 'LCB_sigma': 300.,
                     'RCB_alpha': 7., 'RCB_n': 10., 'RCB_mu': 1500.0, 'RCB_sigma': 320.},
        "brem_two": {'logn_mu': 6.8, 'logn_sigma': 0.42, 'logn_theta': -3200.,
                     'LCB_alpha': 0.54, 'LCB_n': 6., 'LCB_mu': 2100., 'LCB_sigma': 300.,
                     'RCB_alpha': 1., 'RCB_n': 10., 'RCB_mu': 1400.0, 'RCB_sigma': 300.}
    }
}

data_branches = ["J_psi_1S_M", "B_plus_M", "B_plus_DTFM_M", "BDT_score_selection", "e_plus_BremMultiplicity",
                 "e_minus_BremMultiplicity", "L0TISOnly_d", "L0ETOSOnly_d", "J_psi_1S_TRACK_M",
                 'e_plus_L0Calo_ECAL_realET', 'e_minus_L0Calo_ECAL_realET', 'TCKCat', 'e_minus_L0ElectronDecision_TOS',
                 'B_plus_L0Global_TIS', 'e_plus_L0ElectronDecision_TOS', 'e_plus_L0Calo_ECAL_realET',
                 'e_minus_L0Calo_ECAL_realET', 'TCKCat', 'e_plus_L0ElectronDecision_TOS', 'B_plus_L0Global_TIS',
                 'e_minus_L0ElectronDecision_TOS', 'J_psi_1S_ETA',
                 'e_plus_TRACK_P', 'e_plus_P', 'e_minus_TRACK_P', 'e_minus_P',
                 'e_plus_PX', 'e_plus_PY', 'e_plus_PZ', 'e_minus_PX', 'e_minus_PY', 'e_minus_PZ']

mc_branches = ["J_psi_1S_M", "B_plus_M", "B_plus_DTFM_M", "BDT_score_selection", "J_psi_1S_M_TRUE",
               "e_plus_BremMultiplicity", "e_minus_BremMultiplicity", "L0TISOnly_d", "L0ETOSOnly_d",
               'e_plus_L0Calo_ECAL_realET', 'e_minus_L0Calo_ECAL_realET', 'TCKCat', 'e_minus_L0ElectronDecision_TOS',
               'B_plus_L0Global_TIS', 'e_plus_L0ElectronDecision_TOS', 'e_plus_L0Calo_ECAL_realET',
               'e_minus_L0Calo_ECAL_realET', 'TCKCat', 'e_plus_L0ElectronDecision_TOS', 'B_plus_L0Global_TIS',
               'e_minus_L0ElectronDecision_TOS', "J_psi_1S_TRACK_M", "B_plus_M_TRUE", 'J_psi_1S_ETA',
               'e_plus_TRACK_P', 'e_plus_P', 'e_minus_TRACK_P', 'e_minus_P',
               'e_plus_PX', 'e_plus_PY', 'e_plus_PZ', 'e_minus_PX', 'e_minus_PY', 'e_minus_PZ']

hist_dict = {"Jpsi_MC_mee": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                             "bin_width": 30,
                             "num_bins": 39,
                             "x_min": 2430,
                             "x_max": 3600,
                             "x_label": "M(ee), MeV",
                             "y_label": "Events/32 MeV",
                             "x_var": "J_psi_1S_M"},
             "data_mee": {"plot_label": "$B \\rightarrow Kee$",
                          "bin_width": 30,
                          "num_bins": 39,
                          "x_min": 2430,
                          "x_max": 3600,
                          "x_label": "M(ee), MeV",
                          "y_label": "Events/32 MeV",
                          "x_var": "J_psi_1S_M"},
             "Jpsi_MC_mKee": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                              "bin_width": 30,
                              "num_bins": 40,
                              "x_min": 4600,
                              "x_max": 5800,
                              "x_label": "M(Kee), MeV",
                              "y_label": "Events/32 MeV",
                              "x_var": "B_plus_M"},
             "data_mKee": {"plot_label": "$B \\rightarrow Kee$",
                           "bin_width": 30,
                           "num_bins": 40,
                           "x_min": 4600,
                           "x_max": 5800,
                           "x_label": "M(Kee), MeV",
                           "y_label": "Events/32 MeV",
                           "x_var": "B_plus_M"},
             "Jpsi_MC_q2_nobrem": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                                   "bin_width": 70,
                                   "num_bins": 40,
                                   "x_min": 300,
                                   "x_max": 3100,
                                   "x_label": "M(ee) track, MeV",
                                   "y_label": "Events/32 MeV",
                                   "x_var": "q2_nobrem"},
             "q2_nobrem": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                           "bin_width": .2,
                           "num_bins": 70,
                           "x_min": 0.,
                           "x_max": 14.,
                           "x_label": "$q^2$ track, $GeV^2$",
                           "y_label": "Events/.2 $GeV^2$",
                           "x_var": "q2_nobrem"},
             "data_mee_nobrem": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                                 "bin_width": 100,
                                 "num_bins": 30,
                                 "x_min": 300.,
                                 "x_max": 3300.,
                                 "x_label": "M(ee) track, MeV",
                                 "y_label": "Events/100 MeV",
                                 "x_var": "J_psi_1S_TRACK_M"},
             "Jpsi_MC_mee_nobrem": {"plot_label": "$B \\rightarrow KJ/\\psi$",
                                    "bin_width": 100,
                                    "num_bins": 30,
                                    "x_min": 300.,
                                    "x_max": 3300.,
                                    "x_label": "M(ee) track, MeV",
                                    "y_label": "Events/100 MeV",
                                    "x_var": "J_psi_1S_TRACK_M"},
             'eta': {"plot_label": "$B \\rightarrow KJ/\\psi$",
                     "bin_width": .2,
                     "num_bins": 20,
                     "x_min": 2.,
                     "x_max": 6.,
                     "x_label": "J/psi $eta$",
                     "y_label": "Events/0.2 MeV",
                     "x_var": "J_psi_1S_ETA"}
             }
