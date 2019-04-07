# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 11:27:52 2019

@author: Damon
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

"""
def loadData():
    train_data_path = './data_process/data_train.txt'
    train_label_path = './data_process/label_train.txt'
    test_data_path = './data_process/data_test.txt'
    test_label_path = './data_process/label_test.txt'    

    print('Begin loading data...')
    train_X = np.genfromtxt(fname=train_data_path,
							dtype=np.int, max_rows=37322-14929, skip_header=0)
    print('traing_data load done.')
    train_y = np.genfromtxt(fname=train_label_path,
							dtype=np.int, max_rows=37322-14929, skip_header=0)
    print('train_label load done.')
    test_X = np.genfromtxt(fname=test_data_path,
							dtype=np.int, max_rows=14929, skip_header=0)
    print('test_data load done.')
    test_y = np.genfromtxt(fname=test_label_path,
							dtype=np.int, max_rows=14929, skip_header=0)
    print('test_label load done.')
    print('load successfully->\n')

    return train_X, train_y, test_X, test_y
"""

def LoadData():
    data_dir = './data_LDA/LDA_data_'
    dimension = 49
    train_data_path = data_dir + 'n_component_{}.npy'.format(dimension)
    test_data_path  = data_dir + 'n_component_{}_test.npy'.format(dimension)
    
    X_train = np.load(train_data_path)
    X_train = X_train[:, 0:dimension]
    X_test  = np.load(test_data_path)
    X_test  = X_test [:, 0:dimension]
    X = np.concatenate((X_train, X_test))
    
    train_label_path = './data_process/label_train.txt'
    test_label_path = './data_process/label_test.txt'   
    train_y = np.genfromtxt(fname=train_label_path,
							dtype=np.int, max_rows=37322-14929, skip_header=0)
    test_y = np.genfromtxt(fname=test_label_path,
							dtype=np.int, max_rows=14929, skip_header=0)
    print('test_label load done.')
    y = np.concatenate((train_y, test_y))
    
    return X, y

def saveData(save_path, n_component, data):
    np.save(save_path + 'n_component_{}'.format(n_component), data)
    #np.savetxt(save_path + 'n_component_{}_variance'.format(n_component-1), variance )
    #with open (save_path + 'n_component_{}.txt'.format(n_component-1), 'w') as f:
        #f.write("total variance ratio:  ")
        #f.write(str(sum(variance)))

def tSNE_tackle(train_X, n_components):
    #fig = plt.figure('LDA')
    tsne = TSNE(n_components=n_components,verbose=1)
    tsne.fit(train_X)
    X_new = tsne.fit_transform((train_X))
    #print("降维后各主成分的方差值与总方差之比：", tsne.explained_variance_ratio_)
    #print("降维后各主成分的方差值之和：", sum(tsne.explained_variance_ratio_))
    #print("降维前样本数量和维度：",train_X.shape)
    #print("降维后样本数量和维度：",X_new.shape)
    #plt.show()

    return X_new

save_path = './data_tSNE/tSNE_data_'
n_components=2

X, y = LoadData()
SNE_X = tSNE_tackle(X, n_components)
saveData(save_path, n_components, SNE_X)
