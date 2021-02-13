from Service import *
from Fitting import *


obs = zfit.Space('J_psi_1S_M', limits=(2200, 3800))

data = get_data_from_files()
data = categorize_by_brem(data)
data = categorize_by_trig(data)
plot_histograms(data)

for r, data_run in data.items():
    jpsi_sample = data_run["Jpsi_MC"]
    for b, jpsi_brem in jpsi_sample.items():
        for t, jpsi_trig in jpsi_brem.items():

            tag_list = (b, t, r)
            ini_model = create_initial_model(tag_list)
            mc_parameters = initial_fitter(jpsi_trig, ini_model)
            plot_mc_fit_result(ini_model, jpsi_trig["J_psi_1S_M"], tag_list)

            data_trig = data_run["data"][b][t]
            _, fin_model = create_data_fit_model(data_trig, mc_parameters, tag_list)
            plot_data_fit_result(fin_model, ini_model, data_trig["J_psi_1S_M"], tag_list)




