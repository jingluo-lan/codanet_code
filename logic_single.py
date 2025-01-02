#CNN to HLS 

import tensorflow.compat.v2 as tf
# 导入必要的模块和函数
# strip_pruning: 用于移除模型中的剪枝相关参数（如掩码和剪枝的辅助信息）
from tensorflow_model_optimization.sparsity.keras import strip_pruning

# pruning_wrapper: 用于加载剪枝后的模型，确保能够处理剪枝包装的层
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

# _add_supported_quantized_objects: 用于支持 QKeras 量化模型的加载
from qkeras.utils import _add_supported_quantized_objects


# 设置显存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 允许 TensorFlow 在需要时动态分配显存
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        # 异常处理：不能在程序运行后修改 GPU 设置
        print(e)

# 在加载模型时传递自定义损失函数
def focal_loss_fixed(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        # 计算 focal loss
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    
    return focal_loss

from sklearn.metrics import f1_score

def f1_score_metric(y_true, y_pred):
    # 将预测值和真实值转为 0 或 1（假设是二分类任务）
    y_pred = tf.round(y_pred)
    return tf.py_function(f1_score, (y_true, y_pred), tf.float64)


# 将自定义指标传递给 custom_objects
co = {
    'f1_score_metric': f1_score_metric,
    # 如果有其他自定义的损失函数，可以在这里添加
    'focal_loss_fixed': focal_loss_fixed
}

model = tf.keras.models.load_model('/home/ssd1/vscode/nas_hw/models/vggnet-like-hw-2/model_combination_138_conv_blocks_4_filters_8_kernel_4_conv_layers_1_strides_1_dense_8_dropout_0.26.h5', custom_objects=co)

# 使用 strip_pruning 函数移除模型中的剪枝相关参数（如掩码和辅助数据），
# 这样模型在推理时不再依赖这些剪枝信息，从而提高推理效率
model = strip_pruning(model)
model.summary()

import hls4ml  # 导入 hls4ml 库，用于将 Keras 模型转换为可在 FPGA 上运行的 HLS 代码
import plotting  # 导入 plotting 模块，用于打印和可视化配置

# Step 1: 基于 Keras 模型生成 hls4ml 配置
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
# 使用 granularity='name'，为每一层生成一个详细的配置字典。该配置是从 Keras 模型导出的，用于后续的 FPGA 优化。

# Step 2: 设置整个模型的精度和重用因子
# 'ap_fixed<16,6>' 表示定点数类型，16 位宽度，其中 6 位用于整数部分，10 位用于小数部分。
hls_config['Model']['Precision'] = 'ap_fixed<16,6>'

# 设置重用因子 (ReuseFactor)，影响并行度和资源使用率。值为 1 表示不重用任何硬件资源（完全并行）。
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['ReuseFactor'] = 10
# Step 3: 为每一层设置策略和重用因子
# 遍历模型中的所有层，设置每层的策略为 'Latency'（减少推理延迟）
# 'ReuseFactor' 设置为 1，表示所有层都不进行资源重用，允许每一层完全并行执行。
for Layer in hls_config['LayerName'].keys():
    hls_config['LayerName'][Layer]['Strategy'] = 'Latency'  # 设置延迟优先策略
    hls_config['LayerName'][Layer]['ReuseFactor'] = 10 # 设置不重用硬件资源

# 特殊处理 Softmax 输出层
# Softmax 层设置为 'Stable' 策略，这个策略在数值上更加稳定，适合精度要求高的场景（例如分类任务的最终输出）。
# 'Latency' 策略虽然更快，但在 Softmax 层的数值精度上可能不如 'Stable'。
hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'

# 打印 hls4ml 配置，方便查看每一层的设置情况
#plotting.print_dict(hls_config)

# Step 4: 创建 HLS 配置
# 指定使用 Vivado 后端来生成 HLS 代码，Vivado 是 Xilinx FPGA 的开发工具
cfg = hls4ml.converters.create_config(backend='Vitis')

# Step 5: 设置 HLS 转换配置
cfg['IOType'] = 'io_stream'  # 设置 I/O 类型为流式 I/O（对于 CNN 模型必须设置）
cfg['HLSConfig'] = hls_config  # 将生成的 hls 配置传入
cfg['KerasModel'] = model  # 指定要转换的 Keras 模型
cfg['OutputDir'] = './logic_output/bit/vggnet-like-hw-1215/'  # 设置输出目录，HLS 代码和结果将保存到该目录下
#cfg['XilinxPart'] = 'xcu250-figd2104-2L-e'  # 设置目标 Xilinx FPGA 器件型号   
cfg['Part'] = 'xczu9eg-ffvb1156-2-e'                                           # 可选型号：  ZCU102:xczu9eg-ffvb1156-2-e
                                                                                #      PYNQ-Z2:xc7z020clg400-1
                                                                                #      Alevo-U250：xcu250-figd2104-2L-e
# Step 6: 将 Keras 模型转换为 HLS 模型
hls_model = hls4ml.converters.keras_to_hls(cfg)

# Step 7: 编译 HLS 模型
# 编译 HLS 模型，生成用于 FPGA 的 HLS 代码。
hls_model.compile()


import os

# 将 Vivado 的路径添加到系统环境变量 'PATH' 中，以便可以调用 Vivado 工具链

#os.environ['XILINX_VIVADO'] = '/home/ssd0/xilinx/vivado2020.1/Vivado/2020.1'
os.environ['XILINX_VIVADO'] = '/home/ssd0/Vitis2024.1/Vitis_HLS/2024.1'



os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

# 对 hls_model 进行合成，不执行 C 模拟，只进行合成 (synth=True) 和后端合成 (vsynth=True)
hls_model.build(csim=False, synth=True, vsynth=True)
#hls_model.build(csim=False, export=True, bitfile=True)

 