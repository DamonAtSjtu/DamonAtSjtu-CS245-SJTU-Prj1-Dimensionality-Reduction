import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

def getShape(M):
	return (len(M), len(M[0]))


def text_save(filename, data):	# filename为写入CSV文件的路径，data为要写入数据列表.

    file = open(filename,'a')

    for i in range(len(data)):

        s = str(data[i]).replace('[','').replace(']','')	# 去除[],这两行按数据不同，可以选择

        s = s.replace("'",'').replace(',','') +'\n'   	# 去除单引号，逗号，每行末尾追加换行符

        file.write(s)

    file.close()

    print("save sucess")

print('begin loading data')
data = np.genfromtxt(fname='./Data/AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt',
							dtype=np.float, max_rows=37322, skip_header=0)
print('data load done.')
print('begin loading label')
label = np.genfromtxt(fname='./Data/AwA2-features/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt',
							dtype=np.float, max_rows=37322, skip_header=0)
print('label load done.')
label = label.tolist()

for feature_num in [800, 1000, 1200, 1400, 1600, 1800, 2000]:
	# feature_num = 600	#number of reserved features
	test_type = chi2 	#type of test: chi2, f_classif, mutual_info_classif
	processed_data = SelectKBest(test_type, k = feature_num).fit_transform(data,label)
	dim = len(processed_data[0])
	print(dim)

	processed_data = processed_data.tolist()
	# label = label.tolist()
	train_data = []
	train_label = []
	test_data = []
	test_label = []
	k = 1
	for i in range(37322):
		if(k <= 8):
			if(k % 2 == 1):
				train_data.append(processed_data[i])
				train_label.append(label[i])
				k = k+1
			else:
				test_data.append(processed_data[i])
				test_label.append(label[i])
				k = k + 1
		else:
			if(k == 9):
				train_data.append(processed_data[i])
				train_label.append(label[i])
				k = k+1
			else:
				train_data.append(processed_data[i])
				train_label.append(label[i])
				k = 1

	np.savetxt("./Data/data_reduction/data_train"+ "_fs2_"+str(feature_num)+"_"+ "chi2" +".txt", train_data)
	# np.savetxt("./Data/data_reduction/label_train.txt", train_label)
	np.savetxt("./Data/data_reduction/data_test"+"_fs2_"+str(feature_num)+"_"+ "chi2"+".txt", test_data)
	# np.savetxt("./Data/data_reduction/label_test.txt", test_label)