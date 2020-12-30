import numpy as np
from data_load import load_clus_data, normalize_data, shuffle_data
from scipy.cluster.hierarchy import linkage
# import math
# from scipy.cluster.hierarchy import linkage
# # from utils import pdist_node


class ARTParam:
    def __init__(self, net_type, inp_dim, rho, beta, alpha=None):
        self.net_type = net_type
        self.inp_dim = inp_dim
        self.rho = rho
        self.beta = beta
        self.alpha = alpha


class ARTNet:
    def __init__(self, params):
        # Settings
        self.net_type = params.net_type
        self.inp_dim = params.inp_dim
        self.rho = params.rho
        self.beta = params.beta
        if self.net_type == 'ART':
            self.alpha = params.alpha

        # Initialize Network
        if self.net_type == 'ART':
            self.W = []
        elif self.net_type == 'CRN':
            self.W = []
            self.C = []
            self.NoE = []
        self.resonance = False
        self.num_C = 0

    ## Train
    def Train(self, inp):
        num_C = self.num_C
        self.Complement_Coding(inp) # x
        self.Code_Activation() # T, T_check
        self.Category_Choice() # Winner_idx
        ## Template Matching
        if num_C == 0:
            pass
        else:
            m = self.Match_func()
            while m < self.rho: # Resonance fail
                self.T[self.Winner_idx] = -0.001
                self.T_Check[self.Winner_idx] = True
                if self.T_Check == [True] * num_C:
                    break
                self.Category_Choice()
                m = self.Match_func()

            if self.T_Check == [True] * num_C:
                self.resonance = False
            else:
                self.resonance = True
                self.update_idx = self.Winner_idx

        ## Tempalte Learning or Make_new_node
        if self.resonance == True:
            self.Template_Learning()
        else:
            self.Make_new_node()

    ## Test
    def Test(self, I):
        self.Complement_Coding(I)
        self.Code_Activation()
        self.Category_Choice()
        return self.Winner_idx

    ## Complement Coding
    def Complement_Coding(self, inp):
        self.x = np.concatenate((inp, 1 - inp)) #inp is dim-array

    ## Code Activation
    def Code_Activation(self):
        num_C = self.num_C
        if num_C == 0:
            pass
        else:
            self.T = self.Choice_func()
            self.T_Check = [False] * num_C

    ## Category Choice
    def Category_Choice(self):
        num_C = self.num_C
        if num_C == 0:
            pass
        else:
            self.Winner_idx = np.argmax(self.T)

    ## Choice function
    def Choice_func(self):
        if self.net_type == 'ART':
            W = self.W
            x = self.x
            T = np.linalg.norm(np.minimum(W, x), ord=1, axis=1) / (self.alpha + np.linalg.norm(W, ord=1, axis=1))
            return T
        elif self.net_type == 'CRN':
            I = self.x[:self.inp_dim]
            Norm = np.linalg.norm(self.C - I, axis=1)
            T = np.exp(-Norm)
            return T

    ## Match function
    def Match_func(self):
        x = self.x
        Winnder_idx = self.Winner_idx
        W = self.W
        return np.linalg.norm(np.minimum(x, W[Winnder_idx]), ord=1)/np.linalg.norm(x, ord=1)

    ## Make New_Node
    def Make_new_node(self):
        if self.num_C == 0:
            if self.net_type == 'ART':
                self.W = np.expand_dims(self.x.copy(), 0)
            elif self.net_type =='CRN':
                self.W = np.expand_dims(self.x.copy(), 0)
                self.C = np.expand_dims(self.x[:self.inp_dim].copy(), 0)
                self.NoE = np.array([[1]])
            self.num_C = self.num_C + 1
        else:
            if self.net_type =='ART':
                self.W = np.row_stack((self.W, self.x))
            elif self.net_type == 'CRN':
                self.W = np.row_stack((self.W, self.x))
                self.C = np.row_stack((self.C, self.x[:self.inp_dim]))
                self.NoE = np.row_stack((self.NoE, np.array([[1]])))
            self.num_C = self.num_C+1

    ## Template Learning
    def Template_Learning(self):

        x = self.x
        dim = self.inp_dim
        update_idx = self.update_idx

        W_updated = self.W[update_idx]
        self.W[update_idx] = self.beta * (np.minimum(x, W_updated)) + (1 - self.beta) * W_updated

        if self.net_type == 'CRN':
            C_updated = self.C[update_idx]
            NoE_updated = self.NoE[update_idx]
            C_new = (x[:dim] + NoE_updated * C_updated) / (NoE_updated + 1)
            self.C[update_idx] = self.beta * C_new + (1-self.beta)*C_updated
            self.NoE[update_idx] = NoE_updated + 1

    def HA(self, link):
        if self.C.shape[0] == 1:
            return np.asarray(None)
        else:
            Z = linkage(self.C, link)
        return Z


if __name__ == '__main__':
    data_name = 'iris'
    D, L = load_clus_data(data_name)
    D = normalize_data(D)
    D, L = shuffle_data(D, L)

    num_data = D.shape[0]
    inp_dim = D.shape[1]

    art_params = ARTParam('ART', 4, 0.8, 1.0, 0.001)
    crn_params = ARTParam('CRN', 4, 0.8, 1.0)

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

    Z = CRN.HA('single')