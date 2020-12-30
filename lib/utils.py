from scipy import cluster

def Z_2_results(CRN_R, Z, n_clusters):
    cutree = cluster.hierarchy.cut_tree(Z, n_clusters=n_clusters)
    R = []
    for i in range(len(CRN_R)):
        R.append(cutree[int(CRN_R[i]), 0])
    return R
