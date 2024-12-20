

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import re

# 从文本文件读取数据并解析
def parse_data(file_path):
    data = {
        'ConvBlocks': [],       # 卷积块数量
        'ConvLayersPerBlock': [],# 每个块中的卷积层数
        'Kernels': [],          # 卷积核数量
        'KernelSize': [],       # 卷积核大小
        'Stride': [],           # 步长
        'BRAM': [],             # BRAM使用量
        'DSP': [],              # DSP使用量
        'FF': [],               # FF使用量
        'LUT': []               # LUT使用量
    }

    with open(file_path, 'r') as file:
        for line in file:
            # 使用正则表达式提取数据
            match = re.match(
                r"model_combination_\d+_conv_blocks_(\d+)_filters_(\d+)_kernel_(\d+)_conv_layers_(\d+)_strides_(\d+): BRAM=(\d+), DSP=(\d+), FF=(\d+), LUT=(\d+)", 
                line
            )
            if match:
                conv_blocks, filters, kernel_size, conv_layers_per_block, stride, bram, dsp, ff, lut = map(int, match.groups())
                data['ConvBlocks'].append(conv_blocks)
                data['ConvLayersPerBlock'].append(conv_layers_per_block)
                data['Kernels'].append(filters)
                data['KernelSize'].append(kernel_size)
                data['Stride'].append(stride)
                data['BRAM'].append(bram)
                data['DSP'].append(dsp)
                data['FF'].append(ff)
                data['LUT'].append(lut)

    return pd.DataFrame(data)

# 读取数据
df = parse_data('/home/ssd1/vscode/nas_hw/logic_output/vggnet-like-hw/for_logic/performance_estimates.txt')

# 查看数据框的前几行
print(df.head())


print("XGBOOST：")


import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm  # 用于显示进度条


# 定义输入特征和目标变量
X = df[['ConvBlocks', 'ConvLayersPerBlock', 'Kernels', 'KernelSize', 'Stride']]
y_bram = df['BRAM']
y_dsp = df['DSP']
y_ff = df['FF']
y_lut = df['LUT']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y_lut, test_size=0.2, random_state=42)


unique, counts = np.unique(y_lut, return_counts=True)
print(dict(zip(unique, counts)))

# print(X_test)
# print(y_train)
# print(y_test)


dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数
params = {
    'objective': 'reg:squarederror',    # 目标函数
    'eval_metric': 'rmse',              # 使用RMSE作为评估指标
    'max_depth': 6,                     # 树的最大深度
    'learning_rate': 0.01,              # 学习率
    'subsample': 0.7,                   # 子样本比例
    'colsample_bytree': 0.7,            # 每棵树的特征采样比例
    'device': 'cuda',                   # 使用 GPU 进行训练
    #'tree_method': 'hist', 
    #'predictor': 'gpu_predictor',       # 使用 GPU 预测
}

# 创建验证集，用于评估模型
evals = [(dtrain, 'train'), (dtest, 'eval')]


# 训练XGBoost模型并显示进度条
print("Training XGBoost model with GPU...\n")
num_boost_round = 100  # 训练的迭代轮数

# 用来保存训练过程中的loss
train_loss = []
eval_loss = []

# 使用 tqdm 显示训练进度
for i in tqdm(range(num_boost_round), desc="Training Progress", ncols=100):
    model = xgb.train(params, dtrain, num_boost_round=i + 1, evals=evals, verbose_eval=False)
    
    # 获取当前的训练损失
    train_loss.append(model.eval(dtrain))
    eval_loss.append(model.eval(dtest))

# 训练损失和验证损失的记录
train_loss = [float(loss.split(":")[1]) for loss in train_loss]
eval_loss = [float(loss.split(":")[1]) for loss in eval_loss]

# 绘制训练过程中的损失图
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_boost_round + 1), train_loss, label='Training Loss', color='blue')
plt.plot(range(1, num_boost_round + 1), eval_loss, label='Validation Loss', color='red')
plt.xlabel('Boosting Round')
plt.ylabel('Loss (RMSE)')
plt.title('Training and Validation Loss during Training')
plt.legend()
plt.grid(True)
plt.savefig('./nihe/vggnet-like-hw/xgboost_loss.png')

# 预测
y_pred = model.predict(dtest)

# 计算R²和均方误差
r2 = r2_score(y_test, y_pred)
print(f"\nR²: {r2:.4f}")

# 输出预测结果
predictions = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# 计算真实值和预测值差值百分比
predictions['Difference_Percentage'] = abs(predictions['Predicted'] - predictions['Actual']) / predictions['Actual'] * 100

# 打印差值百分比
print("\nPrediction Results with Difference Percentage:")
print(predictions)

# 保存模型
model.save_model('./nihe/vggnet-like-hw/xgboost_lut_model.json')

# --------------------------------------
# 加载保存的模型并进行预测

# 加载保存的模型
loaded_model = xgb.Booster()
loaded_model.load_model('/home/ssd1/vscode/nas_hw/nihe/vggnet-like-hw/best/xgboost_lut_model.json')

# 使用加载的模型进行预测
#dtest = 
y_pred_loaded_model = loaded_model.predict(dtest)

# 计算 R² 和均方误差
r2_loaded = r2_score(y_test, y_pred_loaded_model)
print(f"\nR² of Loaded Model: {r2_loaded:.4f}")

# 输出加载模型的预测结果
predictions_loaded = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_loaded_model})

# 计算真实值和预测值差值百分比
predictions_loaded['Difference_Percentage'] = abs(predictions_loaded['Predicted'] - predictions_loaded['Actual']) / predictions_loaded['Actual'] * 100

# 打印差值百分比
print("\nLoaded Model Prediction Results with Difference Percentage:")
print(predictions_loaded)
