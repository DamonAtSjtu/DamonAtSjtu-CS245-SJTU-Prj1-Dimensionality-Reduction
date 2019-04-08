import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf


DimRedu = 452


def xavier_init(fan_in, fan_out, constant=1):
    # 权重初始化，使得权重满足(low, high)的均匀分布
    # fan_in: 输入节点的数量； fan_out: 输出节点的数量
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random_uniform((fan_in, fan_out), minval=low, maxval=high,
                             dtype=tf.float32)
    
class AdditiveGaussianNoiseAutoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus,
                 optimizer = tf.train.AdamOptimizer(), scale=0.1):
        # 神经网络的构建函数；
        # n_input：输入变量数； n_hidden：隐含层节点数； transfer_function: 隐含层激活函数
        # optimizer: 训练优化器； scale: 高斯噪声系数
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.placeholder(tf.float32)
        self.training_scale = scale
        network_weight = self._initialize_weights()
        self.weights = network_weight

        # 网络结构
        self.x = tf.placeholder(tf.float32,[None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), 
                                                     self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), 
                                               self.weights['b2']) # 不需要使用激活函数

        # 自编码器的损失函数
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.x, self.reconstruction), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)


    def _initialize_weights(self):
        # 参数初始化函数
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
        
        return all_weights

    def partial_fit(self, X):
        # 计算损失以及执行一步训练的函数
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X, 
                                  self.scale: self.training_scale})
        return cost

    def calc_total_cost(self, X):
        # 只求损失的函数
        return self.sess.run(self.cost, feed_dict = {self.x: X, 
                             self.scale: self.training_scale})

    def transform(self, X, _scale):
        # 返回隐含层的输出结果
        # return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: self.training_scalee})
        return self.sess.run(self.hidden, feed_dict = {self.x: X, self.scale: _scale})

    def generate(self, hidden=None):
        # 将提取到的高阶特征复原为原始函数
        if hidden is None:
            hidden = np.random.normal(size=self.weights['b1'])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        # 从原始数据到重建数据的过程
        return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

    def getWeights(self):
        # 获取权w1
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])




def standard_scale(X_train, X_test):
    # 对训练和测试数据进行标准化，需要注意的是必须保证训练集和测试集都使用完全相同的Scale
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

# # 对训练集和测试集进行标准化处理
# X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)
print('begin loading data')
X_train = np.genfromtxt(fname='./data_process(float)/data_train.txt',
                            dtype=np.float, max_rows=37322-14929, skip_header=0)
# X_train = np.genfromtxt(fname='./data/Animals_with_Attributes2/Features/ResNet101/AwA2-features.txt',
#                             dtype=np.float, max_rows=37322, skip_header=0)
X_test = np.genfromtxt(fname='./data_process(float)/data_test.txt',
                            dtype=np.float, max_rows=14929, skip_header=0)
print('data load done.')

n_samples = 22393 # num of train samples
train_epochs = 20   # 最大训练轮数
batch_size = 128    # 每次训练取块的样本数
display_step = 1    # 每个一轮就显示一次损失



dim_range = []
# 创建一个去噪自编码器的实例
autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=2048, n_hidden=DimRedu, transfer_function=tf.nn.softplus, 
                                               optimizer = tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)

for epoch in range(train_epochs):
    avg_cost = 0
    total_batch = int(n_samples / batch_size)   # 总共能够获取的块数
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)   # 获得的每一块的数据
        # train_step
        cost = autoencoder.partial_fit(batch_xs)
        avg_cost += cost / n_samples * batch_size  # 计算获得平均损失
    if epoch % display_step == 0:
        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

# 输出总的测试误差
print("Total cost:" + str(autoencoder.calc_total_cost(X_test)))

trainDat_name = './data_process(float)/autoencoder/autoencoder_dim_' + str(DimRedu) + '_train_data.txt'
testDat_name = './data_process(float)/autoencoder/autoencoder_dim_' + str(DimRedu) + '_test_data.txt'

print('begin saving data.')
new_train_data = autoencoder.transform(X_train, _scale=0.01)
np.savetxt(trainDat_name, new_train_data)
print('train data save successfully.')

new_test_data = autoencoder.transform(X_test, _scale=0.01)
np.savetxt(testDat_name, new_test_data)
print('test data save successfully.')
