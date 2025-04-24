import os
# 设置环境变量以便调用 Vivado 工具链
os.environ['XILINX_VIVADO'] = '/home/ssd0/xilinx/vivado2019.2/Vivado/2019.2'
os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']


import tensorflow as tf
import hls4ml
print(hls4ml.converters.get_supported_keras_layers())

from tensorflow.keras.utils import get_custom_objects

from qkeras.utils import _add_supported_quantized_objects

co = {}
_add_supported_quantized_objects(co)


# 确保GPU可以按需增长
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

#加载已训练的 TensorFlow 模型
model = tf.keras.models.load_model(
    './checkpoints/best_iou_model.h5',
    custom_objects=co  # 确保自定义被加载（损失函数、评价指标、量化层等等）
    
    )

model.summary()
for layer in model.layers:
    print(f"Layer name: {layer.name}, Layer type: {type(layer)}")

# 创建 HLS 转换器配置
hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# 修改配置（如果需要）
hls_config['Model']['Precision'] = 'ap_fixed<8,3>' 
hls_config['Model']['Strategy'] = 'Resource'
hls_config['Model']['ReuseFactor'] = 64  # 重用因子设置

for Layer in hls_config['LayerName'].keys():
    hls_config['LayerName'][Layer]['Strategy'] = 'Resource'  # 设置优先策略  Latency  or  Area   Resource
    hls_config['LayerName'][Layer]['ReuseFactor'] = 64  # 设置重用硬件资源
    hls_config['LayerName'][Layer]['Precision'] = 'ap_fixed<8,3>'


# print("-----------------------------------")
# import plotting
# plotting.print_dict(hls_config)
# print("-----------------------------------")


cfg = hls4ml.converters.create_config(backend='VivadoAccelerator')

# Step 5: 设置 HLS 转换配置
cfg['IOType'] = 'io_stream'  # 设置 I/O 类型为流式 I/O（对于 CNN 模型必须设置）
cfg['HLSConfig'] = hls_config  # 将生成的 hls 配置传入
cfg['KerasModel'] = model  # 指定要转换的 Keras 模型
cfg['OutputDir'] = './logic_results/bit/zcu102-1'# 设置输出目录，HLS 代码和结果将保存到该目录下  
cfg['AcceleratorConfig']['Board']='zcu102'
                                             # 可选型号：pynq-z2 : xc7z020clg400-1
                                             #          zcu102 : xczu9eg-ffvb1156-2-e
                                             #       alveo-u50 : xcu50-fsvh2104-2-e
                                             #       alveo-u250 : xcu250-figd2104-2L-e
                                             #       alveo-u200 : xcu200-fsgd2104-2-e
# 将模型转换为 HLS 代码
hls_model = hls4ml.converters.keras_to_hls(cfg)


# 保存 HLS 代码到文件
hls_model.compile()

#生成比特流文件
hls_model.build(csim=False, export=True, bitfile=True)