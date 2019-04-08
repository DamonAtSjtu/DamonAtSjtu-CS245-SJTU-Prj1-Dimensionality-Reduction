import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

def getShape(M):
	return (len(M), len(M[0]))


# # ori_data
# train_data_path = './data_process/data_train.txt'
# train_label_path = './data_process/label_train.txt'
# test_data_path = './data_process/data_test.txt'
# test_label_path = './data_process/label_test.txt'

# dimRed data for k-ford with dim equals to 50
train_data_path = './data_process(float)/pca/pca_dim_50_train_data.txt'
train_label_path = './data_process(float)/pca/pca_dim_50_train_label.txt'
test_data_path = './data_process(float)/pca/pca_dim_50_test_data.txt'
test_label_path = './data_process(float)/pca/pca_dim_50_test_label.txt'

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



k_scores = []
# set your C range here
c_range = [0.0021, 0.0023, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003, 0.0032, 0.0035]

for c in c_range:

	svm = SVC(kernel='linear', C=c)

	# K-ford
	scores = cross_val_score(svm, train_X, train_y, cv=5, scoring='accuracy')

	k_scores.append(scores.mean())
	print('Cross-Validation (C = ', c, ')done. score = ', scores.mean())
# graph drawing
plt.plot(c_range, k_scores)
plt.xlabel('Value of C for SVM')
plt.ylabel('Cross-Validated Accuracy')
plt.show()
# accu = clf.score(test_X, test_y)
# print('accuracy: ', accu)