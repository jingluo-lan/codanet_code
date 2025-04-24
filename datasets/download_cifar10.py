import tensorflow as tf
import scipy.io
import numpy as np

# 下载 CIFAR-10 数据集
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# 保存数据为 .mat 格式
train_data = {
    'data': x_train,
    'labels': y_train
}

test_data = {
    'data': x_test,
    'labels': y_test
}

# 保存到指定路径
train_path = '/home/ssd1/vscode/nas_hw/datasets/cifar10_train_32x32.mat'
test_path = '/home/ssd1/vscode/nas_hw/datasets/cifar10_test_32x32.mat'

scipy.io.savemat(train_path, train_data)
scipy.io.savemat(test_path, test_data)

print("CIFAR-10 数据已保存为 .mat 格式")


import scipy.io

# 加载训练数据
train_data = scipy.io.loadmat('/home/ssd1/vscode/nas_hw/datasets/cifar10_train_32x32.mat')
test_data = scipy.io.loadmat('/home/ssd1/vscode/nas_hw/datasets/cifar10_test_32x32.mat')

# 获取数据和标签
x_train = train_data['data']
y_train = train_data['labels']
x_test = test_data['data']
y_test = test_data['labels']

# 查看数据形状
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)


# 打印键名，查看结构
print(train_data.keys())

# 检查训练图像数据和标签是否存在
if 'data' in train_data:
    print("训练图像数据：", train_data['data'].shape)
if 'labels' in train_data:
    print("训练标签数据：", train_data['labels'].shape)
