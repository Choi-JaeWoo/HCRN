import numpy as np
import csv
import pandas as pd


def load_clus_data(data_name):

    name_list = ['covtype','iris','wine','yeast','glass','vehicle','segment','bioeg','tae',
                 'balancescale','bloodtransfusion', 'banknote','car','liverdisorder']
    if not data_name in name_list:
        print("Unavailable Data")
        return None


    if data_name == 'covtype':
        file_name = './Data/Clustering_Data/covtype.data'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,
        D = np.array(D)
        L = D[:, D.shape[1] - 1]
        D = D[:, 0:D.shape[1] - 1]
        D = D.astype(np.float)
        L = L.astype(np.int)

        return D, L


    elif data_name == 'iris':
        file_name = './Clustering_Data/iris.data'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,
        D = np.array(D)
        D = D[:, 0:D.shape[1] - 1]
        L = np.zeros(D.shape[0])
        L[0:50] = 1
        L[50:100] = 2
        L[100:150] = 3
        D = D.astype(np.float)
        L = L.astype(np.int)

        return D, L

    elif data_name == 'wine':
        file_name = './Data/Clustering_Data/wine.data'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,
        D = np.array(D)
        L = D[:, 0]
        D = D[:, 1:D.shape[1]]
        D = D.astype(np.float)
        L = L.astype(np.int)

        return D, L

    elif data_name == 'yeast':
        file_name = './Data/Clustering_Data/yeast.data'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split())  # split with ,
        D = np.array(D)
        L = D[:, D.shape[1] - 1]
        D = D[:, 1:D.shape[1] - 1]
        for i in range(L.shape[0]):
            if L[i] == 'CYT':
                L[i] = 1
            elif L[i] == 'NUC':
                L[i] = 2
            elif L[i] == 'MIT':
                L[i] = 3
            elif L[i] == 'ME3':
                L[i] = 4
            elif L[i] == 'ME2':
                L[i] = 5
            elif L[i] == 'ME1':
                L[i] = 6
            elif L[i] == 'EXC':
                L[i] = 7
            elif L[i] == 'VAC':
                L[i] = 8
            elif L[i] == 'POX':
                L[i] = 9
            elif L[i] == 'ERL':
                L[i] = 10
        D = D.astype(np.float)
        L = L.astype(np.int)

        return D, L

    elif data_name == 'glass':
        file_name = './Data/Clustering_Data/glass.data'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,
        D = np.array(D)
        L = D[:, -1]
        D = D[:, 1:-1]

        D = D.astype(np.float)
        L = L.astype(np.int)
        L = np.where(L == 5, 4, L)
        L = np.where(L == 6, 5, L)
        L = np.where(L == 7, 6, L)
        return D, L

    elif data_name == 'vehicle':
        for ch in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
            file_name = './Data/Clustering_Data/vehicle/xa' + ch + '.dat'
            with open(file_name) as file:
                lines = file.readlines()

            for i in range(len(lines)):
                lines[i] = lines[i][:-1]  # Delete \n
            for i in range(len(lines)):
                if lines[i][-1] == ' ':
                    lines[i] = lines[i][:-1]  # Delete \n

            Data = []
            for i in range(len(lines)):
                Data.append(lines[i].split(' '))  # split with ,
            Data = np.array(Data)
            Label = Data[:, -1]
            Label = np.where(Label == 'opel', 1, Label)
            Label = np.where(Label == 'saab', 2, Label)
            Label = np.where(Label == 'bus', 3, Label)
            Label = np.where(Label == 'van', 4, Label)
            Data = Data[:, :-1]
            Data = Data.astype(np.float)
            Label = Label.astype(np.int)

            if ch == 'a':
                D = Data
                L = Label
            else:
                D = np.concatenate((D, Data), axis=0)
                L = np.concatenate((L, Label), axis=0)

        return D, L

    elif data_name == 'segment':
        file_name = './Data/Clustering_Data/segment.dat'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(' '))  # split with ,
        D = np.array(D)
        L = D[:, -1]
        D = D[:, :-1]
        D = D.astype(np.float)
        L = L.astype(np.int)

        return D, L

    elif data_name == 'bioeg':
        file_name = './Data/Clustering_Data/biodeg.csv'
        f = open(file_name, 'r', encoding='utf-8')
        lines = []
        rdr = csv.reader(f)
        for line in rdr:
            lines.append(line)
        f.close()

        for i in range(len(lines)):
            lines[i] = lines[i][0]

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(';'))  # split with ,
        D = np.array(D)
        L = D[:, -1]
        D = D[:, :-1]
        D = D.astype(np.float)
        L = np.where(L == 'RB', 1, L)
        L = np.where(L == 'NRB', 2, L)
        L = L.astype(np.int)

        return D, L

    elif data_name == 'tae':
        file_name = './Data/Clustering_Data/tae.data'
        with open(file_name) as file:
            lines = file.readlines()
        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,
        D = np.array(D)
        L = D[:, -1]
        D = D[:, :-1]
        D = D.astype(np.float)
        L = L.astype(np.int)

        return D, L



    elif data_name == 'balancescale':
        file_name = './Data/Clustering_Data/Balance_Scale/balance-scale.data'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,
        D = np.array(D)
        L = D[:,0]
        for i in range(L.shape[0]):
            if L[i] == 'L':
                L[i] = 1
            elif L[i] == 'B':
                L[i] = 2
            elif L[i] == 'R':
                L[i] = 3
        D = D[:,1:5]
        D = D.astype(np.float)
        L = L.astype(np.int)

        return D,L

    elif data_name == 'bloodtransfusion':
        file_name = './Data/Clustering_Data/Blood_Transfusion/transfusion.data'
        with open(file_name) as file:
            lines = file.readlines()
        lines = lines[1:] # remove description

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,

        D = np.array(D)

        L = D[:,4]
        D = D[:,:4]

        D = D.astype(np.float)
        L = L.astype(np.int)
        return D,L

    elif data_name == 'banknote':
        file_name = './Data/Clustering_Data/data_banknote_authentication.txt'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,

        D = np.array(D)

        L = D[:,4]
        D = D[:,:4]

        D = D.astype(np.float)
        L = L.astype(np.int)
        L = L+1
        return D,L


    elif data_name == 'car':
        file_name = './Data/Clustering_Data/car.data'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,

        D = np.array(D)

        D[:,0:2] = np.where(D[:,0:2] == 'vhigh', 1, D[:,0:2])
        D[:, 0:2] = np.where(D[:,0:2] == 'high', 2, D[:,0:2])
        D[:, 0:2] = np.where(D[:,0:2] == 'med', 3, D[:,0:2])
        D[:, 0:2] = np.where(D[:,0:2] == 'low', 4, D[:,0:2])

        D[:, 2] = np.where(D[:, 2] == '5more', 5, D[:, 2])

        D[:, 3] = np.where(D[:, 3] == 'more', 6, D[:, 3])

        D[:, 4] = np.where(D[:, 4] == 'small', 1, D[:, 4])
        D[:, 4] = np.where(D[:, 4] == 'med', 2, D[:, 4])
        D[:, 4] = np.where(D[:, 4] == 'big', 3, D[:, 4])

        D[:, 5] = np.where(D[:, 5] == 'low', 1, D[:, 5])
        D[:, 5] = np.where(D[:, 5] == 'med', 2, D[:, 5])
        D[:, 5] = np.where(D[:, 5] == 'high', 3, D[:, 5])

        D[:, 6] = np.where(D[:, 6] == 'unacc', 1, D[:, 6])
        D[:, 6] = np.where(D[:, 6] == 'acc', 2, D[:, 6])
        D[:, 6] = np.where(D[:, 6] == 'good', 3, D[:, 6])
        D[:, 6] = np.where(D[:, 6] == 'vgood', 4, D[:, 6])

        L = D[:, 6]
        D = D[:,:6]


        D = D.astype(np.float)
        L = L.astype(np.int)

        return D,L

    elif data_name == 'liverdisorder':
        file_name = './Data/Clustering_Data/bupa.data'
        with open(file_name) as file:
            lines = file.readlines()

        for i in range(len(lines)):
            lines[i] = lines[i][:-1]  # Delete \n

        D = []
        for i in range(len(lines)):
            D.append(lines[i].split(','))  # split with ,

        D = np.array(D)

        L = D[:,6]
        D = D[:,:6]

        D = D.astype(np.float)
        L = L.astype(np.int)
        return D,L

def normalize_data(D):
    dim_data = D.shape[1]
    min_data = np.min(D,0) # (dim)
    max_data = np.max(D,0) # (dim)
    Denom = max_data-min_data
    for i in range(dim_data):
        D[:,i] = (D[:,i]-min_data[i]) /Denom[i]

    return D

def shuffle_data(D,L):
    num_data = D.shape[0]
    idx = np.random.permutation(num_data)
    return D[idx], L[idx]


def load_ai(data_path):
    f = open(data_path)
    lines = f.readlines()
    D = []
    for line in lines:
        data = line.split()
        data = np.asarray(data)
        data = data.astype(np.float)
        D.append(data)
    D = np.asarray(D)
    L = D[:, -1].astype(int)
    D = D[:, :-1]
    return D, L

def load_ai_latent(data_path, label_path):
    D = np.load(data_path)
    L = np.load(label_path)
    D = D + np.random.normal(0, 0.01, D.shape)
    return D, L


def load_iu(data_path):
    f0 = pd.read_excel(data_path)
    names = f0.columns.ravel()
    for i in range(12):
        if i==0:
            col = np.array(f0[names[i]].tolist())+np.random.normal(0,0.1,225)
            col = np.expand_dims(col, axis=1)
        else:
            temp = np.array(f0[names[i]].tolist())+np.random.normal(0,0.1,225)

            temp = np.expand_dims(temp, axis=1)
            col = np.concatenate((col,temp), axis=1)
    for i in range(13,17):
        temp = np.array(f0[names[i]].tolist())
        temp = np.expand_dims(temp, axis=1)
        col = np.concatenate((col, temp), axis=1)

    col = np.random.permutation(col) # Only shuffle along axis=0. Remaining axis not shuffled.
    raw_data = col[:,:12]
    lab = col[:,12:]
    label = np.where(lab==1)[1]
    data = np.expand_dims(raw_data, axis=1)
    data = data[:, 0]
    return data, label

def load_js(data_path):
    f = pd.read_excel(data_path)
    names = f.columns.ravel()
    D = []
    for i in range(names.shape[0]):
        a = np.asarray(f[names[i]].tolist())
        D.append(a)
    D = np.asarray(D)
    D = D.astype('float')
    D = D.swapaxes(0,1)
    L = D[:, 11]
    return D, L

def load_yh(data_path):
    D = np.load(data_path)
    L = D[:,-1]
    D = D[:,:-1]
    return D,L