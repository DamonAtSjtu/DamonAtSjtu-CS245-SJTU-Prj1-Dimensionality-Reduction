import numpy as np


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
data = np.genfromtxt(fname='./data/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt',
							dtype=np.float, max_rows=37322, skip_header=0)
print('data load done.')
print('begin loading label')
label = np.genfromtxt(fname='./data/Animals_with_Attributes2/Features/ResNet101/AwA2-labels.txt',
							dtype=np.float, max_rows=37322, skip_header=0)
print('label load done.')

data = data.tolist()
label = label.tolist()
train_data = []
train_label = []
test_data = []
test_label = []
k = 1
for i in range(37322):
	if(k <= 8):
		if(k % 2 == 1):
			train_data.append(data[i])
			train_label.append(label[i])
			k = k+1
		else:
			test_data.append(data[i])
			test_label.append(label[i])
			k = k + 1
	else:
		if(k == 9):
			train_data.append(data[i])
			train_label.append(label[i])
			k = k+1
		else:
			train_data.append(data[i])
			train_label.append(label[i])
			k = 1



text_save("./data_process(float)/data_train.txt", train_data)
text_save("./data_process(float)/label_train.txt", train_label)
text_save("./data_process(float)/data_test.txt", test_data)
text_save("./data_process(float)/label_test.txt", test_label)
