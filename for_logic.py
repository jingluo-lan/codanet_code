import tensorflow.compat.v2 as tf
from tensorflow_model_optimization.sparsity.keras import strip_pruning
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper
from qkeras.utils import _add_supported_quantized_objects
import hls4ml
import plotting
from sklearn.metrics import f1_score
import os
import glob
import concurrent.futures
from tqdm import tqdm

# 设置 GPU 显存增长
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
    tf.config.experimental.set_memory_growth(device, True)

# 自定义 focal loss 损失函数
def focal_loss_fixed(gamma=2., alpha=0.25):
    def focal_loss(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
        y_true = tf.cast(y_true, tf.float32)
        
        cross_entropy = -y_true * tf.math.log(y_pred)
        weight = alpha * tf.pow(1 - y_pred, gamma)
        return tf.reduce_mean(weight * cross_entropy)
    
    return focal_loss

# 自定义 F1 score 指标
def f1_score_metric(y_true, y_pred):
    y_pred = tf.round(y_pred)
    return tf.py_function(f1_score, (y_true, y_pred), tf.float64)

# 自定义对象字典，加载模型时使用
custom_objects = {
    'f1_score_metric': f1_score_metric,
    'focal_loss_fixed': focal_loss_fixed
}

# 定义模型文件路径模板
model_path_template = '/home/ssd1/vscode/nas_hw/models/vggnet-like-hw/model_combination_*_conv_blocks_*_filters_*_kernel_*_conv_layers_*_strides_*.h5'

# 初始化一个列表用于记录成功综合的模型路径
successful_models = []

# 获取文件路径并按顺序排序
model_paths = sorted(glob.glob(model_path_template))
total_models = len(model_paths)  # 获取模型总数
completed_count = 0  # 已综合模型计数

# 定义综合函数
def synthesize_model(model_path):
    global completed_count  # 使用全局变量来跟踪完成的模型数量

    # 设置输出路径为统一目录下的子文件夹
    output_dir = f"/home/ssd1/vscode/nas_hw/logic_output/vggnet-like-hw/for_logic/{os.path.basename(model_path).replace('.h5', '')}"
    
    # 输出当前已综合数量和剩余数量
    completed_count += 1
    remaining_count = total_models - completed_count
    print(f"准备综合模型: {model_path}")
    print(f"已综合 {completed_count} 个模型，剩余 {remaining_count} 个模型。")

    # 检查模型是否已经综合，若存在输出目录则跳过该模型
    if os.path.exists(output_dir):
        print(f"模型 {model_path} 已综合，跳过。")
        return

    try:
        # 加载模型并移除剪枝相关参数
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        model = strip_pruning(model)

        # 判断模型参数量是否大于 100000，若是则跳过综合
        param_count = model.count_params()
        if param_count > 100000:
            print(f"模型 {model_path} 参数量 {param_count} 超出限制，跳过综合。")
            return

        print(f"正在处理模型: {model_path} (参数量: {param_count})")
        model.summary()

        # Step 1: 从 Keras 模型生成 hls4ml 配置
        hls_config = hls4ml.utils.config_from_keras_model(model, granularity='name')
        hls_config['Model']['Precision'] = 'ap_fixed<16,6>'
        hls_config['Model']['Strategy'] = 'Resource'
        hls_config['Model']['ReuseFactor'] = 10

        # Step 2: 设置每一层的策略和重用因子
        for layer_name in hls_config['LayerName']:
            hls_config['LayerName'][layer_name]['Strategy'] = 'Latency'
            hls_config['LayerName'][layer_name]['ReuseFactor'] = 10

        # 特殊设置 Softmax 输出层
        hls_config['LayerName']['output_softmax']['Strategy'] = 'Stable'

        # Step 3: 创建 HLS 转换配置，使用 Vitis 后端
        hls_cfg = hls4ml.converters.create_config(backend='Vitis')
        hls_cfg['IOType'] = 'io_stream'  # 设置 I/O 类型为流式 I/O（针对 CNN）
        hls_cfg['HLSConfig'] = hls_config
        hls_cfg['KerasModel'] = model
        hls_cfg['OutputDir'] = output_dir  # 设置模型输出目录
        hls_cfg['Part'] = 'xczu9eg-ffvb1156-2-e'  # 目标 FPGA 器件型号

        # 转换 Keras 模型为 HLS 模型
        hls_model = hls4ml.converters.keras_to_hls(hls_cfg)

        # 编译 HLS 模型
        hls_model.compile()

        # 设置环境变量以便调用 Vivado 工具链
        os.environ['XILINX_VIVADO'] = '/home/ssd0/Vitis2024.1/Vitis_HLS/2024.1'
        os.environ['PATH'] = os.environ['XILINX_VIVADO'] + '/bin:' + os.environ['PATH']

        # 合成 HLS 模型，不执行 C 模拟，进行综合 (synth=True) 和后端综合 (vsynth=False)
        hls_model.build(csim=False, synth=True, vsynth=False)
        print(f"完成模型 {model_path} 的 HLS 代码生成。")

        # 记录成功综合的模型路径
        successful_models.append(model_path)

    except Exception as e:
        print(f"综合模型 {model_path} 时发生错误，跳过该模型。错误信息: {e}")

# 使用 ThreadPoolExecutor 实现并行综合
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:  # 这里可以调整 max_workers 来控制并行线程数量
    futures = [executor.submit(synthesize_model, model_path) for model_path in model_paths]
    for future in concurrent.futures.as_completed(futures):
        try:
            future.result()  # 获取结果或抛出异常
        except Exception as e:
            print(f"线程执行中出现错误: {e}")

# 输出成功综合的模型路径和数量
print("\n综合成功的模型路径列表：")
for model in successful_models:
    print(model)
print(f"\n成功综合的模型总数: {len(successful_models)}")
