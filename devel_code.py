import numpy as np
import json

from sklearn import metrics

from lib.model import ARTParam, ARTNet
from lib.data_load import load_clus_data, normalize_data, shuffle_data
from lib.utils import Z_2_results
from lib.vis import plot_rho_perf

num_iter = 100
rho_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
num_class = 3

if __name__ == '__main__':

    data_name = 'iris'
    D, L = load_clus_data(data_name)
    D = normalize_data(D)

    num_data = D.shape[0]
    inp_dim = D.shape[1]
    perf_rho = {'ART': [], 'CRN': [], 'HCRN': []}
    std_rho = {'ART': [], 'CRN': [], 'HCRN': []}

    for rho in rho_list:
        perf_iter = {'ART': [], 'CRN':[], 'HCRN':[]}
        print(rho)
        for iter in range(num_iter):
            D, L = shuffle_data(D, L)
            print(iter)
            art_params = ARTParam('ART', 4, rho, 1.0, 0.001)
            crn_params = ARTParam('CRN', 4, rho, 1.0)

            ART = ARTNet(art_params)
            CRN = ARTNet(crn_params)

            for i in range(num_data):
                ART.Train(D[i])
                CRN.Train(D[i])

            ART_R = []
            CRN_R = []
            for i in range(num_data):
                ART_R.append(ART.Test(D[i]))
                CRN_R.append(CRN.Test(D[i]))

            perf_iter['ART'].append(metrics.normalized_mutual_info_score(L, ART_R, average_method='arithmetic'))
            perf_iter['CRN'].append(metrics.normalized_mutual_info_score(L, CRN_R, average_method='arithmetic'))

            Z = CRN.HA('complete')
            if Z.any():
                HCRN_R = Z_2_results(CRN_R, Z, num_class)

            if Z.any():
                perf_iter['HCRN'].append(metrics.normalized_mutual_info_score(L, HCRN_R, average_method='arithmetic'))

        perf_rho['ART'].append(np.asarray(perf_iter['ART']).mean())
        perf_rho['CRN'].append(np.asarray(perf_iter['CRN']).mean())
        perf_rho['HCRN'].append(np.asarray(perf_iter['HCRN']).mean())

        std_rho['ART'].append(np.asarray(perf_iter['ART']).std())
        std_rho['CRN'].append(np.asarray(perf_iter['CRN']).std())
        std_rho['HCRN'].append(np.asarray(perf_iter['HCRN']).std())


    with open('./ai_perf.txt', 'w') as file:
        json.dump(perf_rho, file)
    with open('./ai_std.txt', 'w') as file:
        json.dump(std_rho, file)

    plot_rho_perf(rho_list, perf_rho, std_rho)