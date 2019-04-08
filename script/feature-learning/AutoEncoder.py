import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn

dimRed = 512

transform = transforms.Compose([transforms.ToTensor()])
								# transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
								#transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
print('begin loading data.')
dataset_train = np.genfromtxt(fname='./data_process(float)/data_train.txt',
                            dtype=np.float, max_rows=37322-14929, skip_header=0)
dataset_test = np.genfromtxt(fname='./data_process(float)/data_test.txt',
                            dtype=np.float, max_rows=14929, skip_header=0)
print('data load done.')
train_loader = DataLoader(dataset=dataset_train,
						batch_size=128,
						shuffle=True)
test_loader = DataLoader(dataset=dataset_test,
						batch_size=128,
						shuffle=True)


class AutoEncoder(torch.nn.Module):

	def __init__(self):
		super(AutoEncoder, self).__init__()
		self.encoder = nn.Sequential(
			torch.nn.Linear(2048, 1024),
			nn.ReLU(),
			nn.Linear(1024, 512),
			nn.ReLU(),
			nn.Linear(512, dimRed),
			nn.ReLU()
			)
		self.decoder = nn.Sequential(
			nn.Linear(dimRed, 512),
			nn.ReLU(),
			nn.Linear(512, 1024),
			nn.ReLU(),
			nn.Linear(1024, 2048)
			)
	def forward(self, x):
		output = self.encoder(x)
		output = self.decoder(output)
		return output
	# get the output of encoder
	def transform(self, x):
		return self.encoder(x).numpy()

model = AutoEncoder()
# print(model)
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.MSELoss()

epoch_n = 10

for epoch in range(epoch_n):
	running_loss = 0.0
	print('Epoch {}/{}'.format(epoch, epoch_n))
	print('-'*10)
	for step, (X_train, _) in enumerate(train_loader):
		# X_train, _ = data

		# noisy_X_train = X_train + 0.5 * torch.randn(X_train.shape)
		# noisy_X_train = torch.clamp(noisy_X_train, 0., 1.)

		# X_train, noisy_X_train = Variable(X_train.view(-1, 28*28)), Variable(noisy_X_train.view(-1, 28*28))
		train_pred = model(X_train)
		loss = loss_func(train_pred, X_train)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		running_loss += loss.data
		# print(np.shape(loss.data))
	print("Loss is: {:.4f}".format(running_loss/len(dataset_train)))


trainDat_name = './data_process(float)/autoencoder/autoencoder_dim_' + str(dimRed) + '_train_data.txt'
testDat_name = './data_process(float)/autoencoder/autoencoder_dim_' + str(dimRed) + '_test_data.txt'
train_data = model.transform(dataset_train)
np.savetxt(trainDat_name, train_data.numpy())
print('train data save successfully.')
test_data = model.transform(dataset_train)
np.savetxt(testDat_name, train_data.numpy())
print('test data save successfully.')
