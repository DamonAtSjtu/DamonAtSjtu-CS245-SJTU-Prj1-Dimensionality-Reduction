import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def getShape(M):
	return (len(M), len(M[0]))


# ori_data
train_data_path = './data_process/data_train.txt'
train_label_path = './data_process/label_train.txt'
test_data_path = './data_process/data_test.txt'
test_label_path = './data_process/label_test.txt'

# # dimRed data
# train_data_path = './data_process(float)/pca/pca_dim_500_train_data.txt'
# train_label_path = './data_process(float)/pca/pca_dim_500_train_label.txt'
# test_data_path = './data_process(float)/pca/pca_dim_500_test_data.txt'
# test_label_path = './data_process(float)/pca/pca_dim_500_test_label.txt'

print('Begin loading data...')
train_X = np.genfromtxt(fname=train_data_path,
							dtype=np.float, max_rows=37322-14929, skip_header=0)
print('traing_data load done.')
train_y = np.genfromtxt(fname=train_label_path,
							dtype=np.int, max_rows=37322-14929, skip_header=0)
print('train_label load done.')
test_X = np.genfromtxt(fname=test_data_path,
							dtype=np.float, max_rows=14929, skip_header=0)
print('test_data load done.')
test_y = np.genfromtxt(fname=test_label_path,
							dtype=np.int, max_rows=14929, skip_header=0)
print('test_label load done.')
print('load successfully->\n')

clf = SVC(kernel='linear', C=0.0028)


print('######### training start ###########')
clf.fit(train_X, train_y)
print('######### training done  ###########\n')

print('######### testing start ###########')
test_pred = clf.predict(test_X)
# print(test_pred[1], test_y[1])
print('######### testing done  ###########\n')
accu = (test_pred == test_y).sum() / float(len(test_y))


# accu = clf.score(test_X, test_y)
print('accuracy: ', accu)