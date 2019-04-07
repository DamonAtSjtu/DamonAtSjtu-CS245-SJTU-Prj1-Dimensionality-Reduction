import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_classification
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
 
from mpl_toolkits.mplot3d import Axes3D

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

def saveData(save_path, n_component, data, test_data, variance):
    np.save(save_path + 'n_component_{}'.format(n_component), data)
    np.save(save_path + 'n_component_{}_test'.format(n_component), test_data)
    np.savetxt(save_path + 'n_component_{}_variance'.format(n_component), variance )
    with open (save_path + 'n_component_{}.txt'.format(n_component), 'w') as f:
        f.write("total variance ratio:  ")
        f.write(str(sum(variance)))

def LDA_tackle(train_X, train_y, test_X,n_components):
    #fig = plt.figure('LDA')
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    print("n_component",n_components)
    lda.fit(train_X,train_y)
    X_new = lda.transform(train_X)
    X_test = lda.transform(test_X) 
    print("降维后各主成分的方差值与总方差之比：", lda.explained_variance_ratio_)
    print("降维后各主成分的方差值之和：", sum(lda.explained_variance_ratio_))
    print("降维前样本数量和维度：",train_X.shape)
    print("降维后样本数量和维度：",X_new.shape)
    #plt.show() 

    return X_new, X_test, lda.explained_variance_ratio_


save_path = './data_LDA/LDA_data_'
n_components=[5,15,25,35,45]


train_X, train_y, test_X, test_y = loadData()
for n_component in n_components:
    LDA_X, X_test, variance_ratio = LDA_tackle(train_X, train_y, test_X,n_component)
    saveData(save_path, n_component, LDA_X, X_test, variance_ratio)

 
