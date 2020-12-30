from model import ARTParam, ARTNet
from data_load import load_clus_data, normalize_data, shuffle_data, load_ai, load_iu, load_ai_latent
from sklearn import metrics
from utils import Z_2_results
import numpy as np
import json

num_iter = 100
rho_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
num_class = 3
if __name__ == '__main__':

    # data_name = 'iris'
    # D, L = load_clus_data(data_name)
    # D, L = load_ai('/home/rit/21_MIL/HCRN/Clustering_Data/AI28_New/valid_data.txt')
    # D, L = load_iu('/home/rit/21_MIL/HCRN/Clustering_Data/jisu_data.xlsx')
    # D, L = load_ai_latent('./Clustering_Data/AI28012/latent.npy', './Clustering_Data/AI28012/label.npy')
    # D, L = load_ai_train('./Clustering_Data/AI28_New/latent.npy', './Clustering_Data/AI28_New/label.npy')

    D, L = load_ai('/home/rit/21_MIL/HCRN/Clustering_Data/AI28012/valid_data.txt')
    D = normalize_data(D)





    D = D[:10000]
    L = L[:10000]

    num_data = D.shape[0]
    inp_dim = D.shape[1]
    perf_rho = {'ART':[], 'CRN':[], 'HCRN':[]}
    std_rho = {'ART':[], 'CRN':[], 'HCRN':[]}
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
            # print("Vigilance Parameter: {}".format(rho))
            # print(metrics.normalized_mutual_info_score(L, ART_R, average_method='arithmetic'))
            # print(metrics.normalized_mutual_info_score(L, CRN_R, average_method='arithmetic'))
            Z = CRN.HA('complete')
            if Z.any():
                HCRN_R = Z_2_results(CRN_R, Z, num_class)

            if Z.any():
                perf_iter['HCRN'].append(metrics.normalized_mutual_info_score(L, HCRN_R, average_method='arithmetic'))

            # print("---------------------------------------------------")
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

        # print(metrics.adjusted_rand_score(L, ART_R))
        # print(metrics.adjusted_rand_score(L, CRN_R))
        # print("---------------------------------------------------")