import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tqdm import tqdm

# 从文本文件读取数据并解析
def parse_data(file_path):
    data = {
        'ConvBlocks': [],       
        'ConvLayersPerBlock': [], 
        'Kernels': [],           
        'KernelSize': [],        
        'Stride': [],            
        'Dense': [],             
        'Dropout': [],           
        'BRAM': [],              
        'DSP': [],               
        'FF': [],                
        'LUT': []                
    }

    with open(file_path, 'r') as file:
        for line in file:
            match = re.match(
                r"model_combination_\d+_conv_blocks_(\d+)_filters_(\d+)_kernel_(\d+)_conv_layers_(\d+)_strides_(\d+)_dense_(\d+)_dropout_([\d.]+): BRAM=(\d+), DSP=(\d+), FF=(\d+), LUT=(\d+)", 
                line
            )
            if match:
                conv_blocks, filters, kernel_size, conv_layers_per_block, stride, dense, dropout, bram, dsp, ff, lut = match.groups()
                data['ConvBlocks'].append(int(conv_blocks))
                data['ConvLayersPerBlock'].append(int(conv_layers_per_block))
                data['Kernels'].append(int(filters))
                data['KernelSize'].append(int(kernel_size))
                data['Stride'].append(int(stride))
                data['Dense'].append(int(dense))
                data['Dropout'].append(float(dropout))
                data['BRAM'].append(int(bram))
                data['DSP'].append(int(dsp))
                data['FF'].append(int(ff))
                data['LUT'].append(int(lut))

    return pd.DataFrame(data)

# 读取数据
df = parse_data('/home/ssd1/vscode/nas_hw/logic_output/vggnet-like-hw-2/for_logic/performance_estimates.txt')

# 查看数据框的前几行
print(df.head())

# 定义输入特征和目标变量
X = df[['ConvBlocks', 'ConvLayersPerBlock', 'Kernels', 'KernelSize', 'Stride', 'Dense', 'Dropout']]
y = df['LUT']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 检查 LUT 分布
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique, counts)))

# 将数据转换为XGBoost的DMatrix格式
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# 设置XGBoost参数
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.01,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'tree_method': 'hist',
    'device': 'cuda',
}

# 创建验证集列表
evals = [(dtrain, 'train'), (dtest, 'eval')]

# 训练XGBoost模型
print("Training XGBoost model with GPU...\n")
num_boost_round = 1000

# 创建列表来收集损失
train_loss = []
eval_loss = []

# 创建自定义回调函数，继承TrainingCallback类
from xgboost.callback import TrainingCallback

class LossCollector(TrainingCallback):
    def after_iteration(self, model, epoch, evals_log):
        # 每次迭代后收集损失
        if evals_log:
            if 'train' in evals_log and 'rmse' in evals_log['train']:
                train_loss.append(evals_log['train']['rmse'][-1])
            if 'eval' in evals_log and 'rmse' in evals_log['eval']:
                eval_loss.append(evals_log['eval']['rmse'][-1])
        
        # 每100次迭代打印一次
        if epoch % 100 == 0:
            print(f"[{epoch}] Training RMSE: {train_loss[-1]:.4f}, Validation RMSE: {eval_loss[-1]:.4f}")
        
        # 返回False表示继续训练
        return False

# 使用新的回调类实例
model = xgb.train(
    params, 
    dtrain, 
    num_boost_round=num_boost_round,
    evals=evals,
    callbacks=[LossCollector()],
    verbose_eval=False  # 关闭默认的评估打印，因为我们在回调中自定义了打印
)

# 绘制训练过程中的损失图
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
plt.plot(range(1, len(eval_loss) + 1), eval_loss, label='Validation Loss')
plt.xlabel('Boosting Round')
plt.ylabel('Loss (RMSE)')
plt.title('Training and Validation Loss during Training')
plt.legend()
plt.grid(True)
plt.savefig('./nihe/vggnet-like-hw-2/LUT/xgboost_loss.png')

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
model.save_model('./nihe/vggnet-like-hw-2/LUT/xgboost_model.json')