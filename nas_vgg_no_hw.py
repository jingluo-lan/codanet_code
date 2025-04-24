

import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from skopt import gp_minimize
from skopt.space import Integer, Real
import xgboost as xgb
import pandas as pd

physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# 确保保存模型的目录存在
os.makedirs("models/vggnet-like-test", exist_ok=True)


# 数据加载与预处理
def load_data(train_path, test_path):
    train_data = scipy.io.loadmat(train_path)
    test_data = scipy.io.loadmat(test_path)
    
    train_images = np.transpose(train_data['X'], (3, 0, 1, 2))  # 转换图像维度为 [num_images, height, width, channels]
    train_labels = train_data['y']
    test_images = np.transpose(test_data['X'], (3, 0, 1, 2))
    test_labels = test_data['y']
    
    # 将标签为10的转换为0（假设标签从0到9）
    train_labels[train_labels == 10] = 0
    test_labels[test_labels == 10] = 0
    
    # 标签转换为 one-hot 编码
    train_labels = to_categorical(train_labels, num_classes=10)
    test_labels = to_categorical(test_labels, num_classes=10)
    
    # 数据归一化
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    
    # 划分训练集和验证集（例如，90%用于训练，10%用于验证）
    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.1, random_state=42)
    
    # 输出各数据集的大小
    print("训练集大小:", len(train_images))
    print("验证集大小:", len(val_images))
    print("测试集大小:", len(test_images))
    
    # 创建 TensorFlow 数据集
    batch_size = 256
    ds_full_train = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    ds_val = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
    ds_test = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    
    # 数据增强函数
    def preprocess_image(image, label):
        image = tf.cast(image, tf.float32)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        return image, label
    
    # 应用数据增强并批处理
    ds_train = ds_full_train.map(preprocess_image).shuffle(1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    
    for images, labels in ds_train.take(1):
        print("ds_train image shape:", images.shape)
        print("ds_train label shape:", labels.shape)
    
    for images, labels in ds_val.take(1):
        print("ds_val image shape:", images.shape)
        print("ds_val label shape:", labels.shape)
    
    for images, labels in ds_test.take(1):
        print("ds_test image shape:", images.shape)
        print("ds_test label shape:", labels.shape)
    
    return ds_train, ds_val, ds_test, train_images.shape[1:]

# 定义 Focal Loss
def focal_loss(gamma=2., alpha=0.25):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        
        loss = alpha * tf.pow(1. - p_t, gamma) * cross_entropy
        return tf.reduce_sum(loss, axis=1)
    
    return focal_loss_fixed

# 定义 F1-Score 作为 Keras 指标
def f1_score_metric(y_true, y_pred):
    y_pred = tf.argmax(y_pred, axis=-1)
    y_true = tf.argmax(y_true, axis=-1)
    
    # 使用 sklearn 的 f1_score 函数来计算 F1 score，设置为 multiclass 计算
    f1 = tf.py_function(func=lambda yt, yp: f1_score(yt, yp, average='macro'), inp=[y_true, y_pred], Tout=tf.float32)
    
    return f1

# 构建模型
def build_model(conv_blocks, filters, kernel_size, dense_units, dropout_rate, conv_layers_per_block, strides, initial_lr, input_shape):
    model = models.Sequential()
    
    # 第一个卷积块
    model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu',
                            input_shape=input_shape, padding='same', strides=(strides, strides), name="conv_block1_1"))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu',
                            padding='same', strides=(strides, strides), name="conv_block1_2"))
    model.add(layers.BatchNormalization())

    # 后续卷积块
    for block in range(conv_blocks - 1):
        for layer in range(conv_layers_per_block):
            model.add(layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size), activation='relu',
                                    strides=(strides, strides), padding='same', name=f"conv_block_{block+1}_layer_{layer+1}"))
            model.add(layers.BatchNormalization())

        # 池化层
        if model.output_shape[1] > 1 and model.output_shape[2] > 1:
            model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same', name=f"maxpool_block_{block+1}"))
        else:
            print(f"Skipping MaxPooling2D at block {block+1} due to small input size")

    model.add(layers.Flatten())
    model.add(layers.Dense(dense_units, activation='relu', name="dense1"))
    model.add(layers.Dropout(dropout_rate, name="dropout1"))
    model.add(layers.Dense(10, activation='softmax', name="output_softmax"))

    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=initial_lr),
        loss=focal_loss(gamma=2., alpha=0.25),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), f1_score_metric]
    )

    #model.summary()
    return model

# 计算模型参数量
def calculate_model_params(model):
    total_params = 0
    for layer in model.layers:
        total_params += layer.count_params()
    return total_params

# 目标函数，贝叶斯优化时调用
def objective(params):
    global best_accuracy, best_f1, best_auc
    global best_model_path, best_f1_model_path, best_auc_model_path
    global counter
    
    conv_blocks, filters_exp, kernel_size, dense_units_exp, dropout_rate, conv_layers_per_block, strides = params
    counter += 1
    print("-------------------------------------------------------------------------------------------")
    print(f"当前尝试次数：{counter}")

    filters = 2 ** filters_exp
    dense_units = 2 ** dense_units_exp

    print(f"当前超参数组合: conv_blocks={conv_blocks}, filters={filters}, kernel_size={kernel_size}, "
          f"dense_units={dense_units}, dropout_rate={dropout_rate}, conv_layers_per_block={conv_layers_per_block}, "
          f"strides={strides}")

    # 构建模型
    model = build_model(conv_blocks, filters, kernel_size, dense_units, dropout_rate, conv_layers_per_block, strides, initial_lr, input_shape)

    # 计算参数量并判断是否跳过
    total_params = calculate_model_params(model)
    print("当前组合模型参数量", total_params)
    if total_params > 2000000:
        print(f"超参数组合: 参数量 {total_params} 超过限制，跳过该组合")
        return 5000  # 返回一个较高的损失以跳过该组合

    # 设置早期停止和学习率调度回调
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=0, min_lr=1e-9)

    # 训练模型
    model.fit(ds_train, epochs=100, verbose=0, callbacks=[early_stopping, lr_scheduler], validation_data=ds_val)

    # 在测试集上评估模型
    results = model.evaluate(ds_test, verbose=1)
    # 根据模型编译时的指标顺序
    loss, accuracy, auc, f1 = results
    print(f"当前超参数组合的测试集准确率: {accuracy:.4f}, AUC: {auc:.4f}, F1-Score: {f1:.4f}")

    # 保存当前配置的模型，并在文件名中加入dense_units和dropout_rate
    current_model_path = (f"models/vggnet-like-test/model_combination_{counter}_conv_blocks_{conv_blocks}"
                          f"_filters_{filters}_kernel_{kernel_size}_conv_layers_{conv_layers_per_block}"
                          f"_strides_{strides}_dense_{dense_units}_dropout_{dropout_rate:.2f}.h5")
    model.save(current_model_path)
    print(f"已保存当前配置的模型到 {current_model_path}")

    # 如果当前模型准确率优于之前的最佳准确率，另存为最佳准确率模型
    if accuracy > best_accuracy:
        print(f"新最佳准确率模型! 测试集准确率从 {best_accuracy:.4f} 提升到 {accuracy:.4f}，正在另存为最佳模型...")
        best_accuracy = accuracy
        model.save(best_model_path)
        print(f"已保存最佳准确率模型到 {best_model_path}")

    # 如果当前模型 F1 分数优于之前的最佳 F1 分数，另存为最佳 F1 模型
    if f1 > best_f1:
        print(f"新最佳 F1 模型! F1 Score 从 {best_f1:.4f} 提升到 {f1:.4f}，正在另存为最佳 F1 模型...")
        best_f1 = f1
        model.save(best_f1_model_path)
        print(f"已保存最佳 F1 模型到 {best_f1_model_path}")

    # 如果当前模型 AUC 优于之前的最佳 AUC，另存为最佳 AUC 模型
    if auc > best_auc:
        print(f"新最佳 AUC 模型! AUC 从 {best_auc:.4f} 提升到 {auc:.4f}，正在另存为最佳 AUC 模型...")
        best_auc = auc
        model.save(best_auc_model_path)
        print(f"已保存最佳 AUC 模型到 {best_auc_model_path}")

    # 将数据保留到小数点后四位，并保存到CSV文件
    with open(output_csv_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            counter, conv_blocks, filters, kernel_size, dense_units, 
            f"{dropout_rate:.4f}", conv_layers_per_block, strides, total_params, 
            f"{loss:.4f}", f"{accuracy:.4f}", f"{best_accuracy:.4f}", f"{f1:.4f}",
            f"{best_f1:.4f}", f"{auc:.4f}", f"{best_auc:.4f}", 
        ])

    # 优化目标，这里选择优化 AUC
    return -auc  # 返回负的 AUC 以进行最小化


# 贝叶斯优化的搜索空间
search_space = [
    Integer(1, 5, name='conv_blocks'),        # 卷积块的个数
    Integer(1, 6, name='filters_exp'),        # 滤波器数量的指数
    Integer(2, 4, name='kernel_size'),        # 卷积核的大小
    Integer(1, 7, name='dense_units_exp'),    # 全连接层的神经元数量的指数
    Real(0.2, 0.5, name="dropout_rate"),      # Dropout率
    Integer(1, 4, name='conv_layers_per_block'),  # 每个卷积块的卷积层个数
    Integer(1, 3 , name='strides')             # 步长
]

# 全局变量用于记录最佳模型
best_accuracy = -np.inf
best_f1 = -np.inf
best_auc = -np.inf

best_model_path = "./vggnet-like-test/best_model_overall.h5"
best_f1_model_path = "./vggnet-like-test/best_f1_model.h5"
best_auc_model_path = "./vggnet-like-test/best_auc_model.h5"

# 初始化 CSV 文件和表头
import csv
output_csv_path = './vggnet-like-test/hyperparameter_search_results.csv'
with open(output_csv_path, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        'Trial Number', 'Conv Blocks', 'Filters', 'Kernel Size', 'Dense Units',
        'Dropout Rate', 'Conv Layers per Block', 'Strides', 'Total Params',
        'Test Loss', 'Test Accuracy', 'Best_accuracy', 'F1 Score',
        'Best_F1 Score', 'AUC', 'Best_AUC',
    ])

counter = 0  # 用于计数

initial_lr = 0.001  # 固定初始学习率（不在搜索空间内）

# 加载数据集
train_path = '/home/ssd1/vscode/hls4ml-tutorial-main/datasets/train_32x32.mat'
test_path = '/home/ssd1/vscode/hls4ml-tutorial-main/datasets/test_32x32.mat'

ds_train, ds_val, ds_test, input_shape = load_data(train_path, test_path)

# 使用贝叶斯优化进行超参数搜索
result = gp_minimize(objective, search_space, n_calls=400, random_state=42)

# 打印当前已找到的超参数组合（贝叶斯优化之后）
print("\n当前已找到的超参数组合：")
for i, params in enumerate(result.x_iters):
    conv_blocks, filters_exp, kernel_size, dense_units_exp, dropout_rate, conv_layers_per_block, strides = params
    filters = 2 ** filters_exp
    dense_units = 2 ** dense_units_exp
    print(f"第 {i+1} 次尝试: conv_blocks={conv_blocks}, filters={filters}, kernel_size={kernel_size}, "
          f"dense_units={dense_units}, dropout_rate={dropout_rate:.2f}, conv_layers_per_block={conv_layers_per_block}, strides={strides}")

# 输出最优超参数
best_params = result.x
best_conv_blocks, best_filters_exp, best_kernel_size, best_dense_units_exp, best_dropout_rate, best_conv_layers_per_block, best_strides = best_params
best_filters = 2 ** best_filters_exp
best_dense_units = 2 ** best_dense_units_exp

print("\n最优超参数:")
print(f"conv_blocks={best_conv_blocks}")
print(f"filters={best_filters}")
print(f"kernel_size={best_kernel_size}")
print(f"dense_units={best_dense_units}")
print(f"dropout_rate={best_dropout_rate:.2f}")
print(f"conv_layers_per_block={best_conv_layers_per_block}")
print(f"strides={best_strides}")

print(f"\n最优 AUC: { -result.fun:.4f}")


