import os
import xml.etree.ElementTree as ET

# 定义要遍历的根目录
root_dir = '/home/ssd1/vscode/nas_hw/logic_output/vggnet-like-hw-2/for_logic'

# 定义要读取的文件路径模板
file_template = 'myproject_prj/solution1/syn/report/csynth.xml'

# 存储提取的信息
performance_data = []

# 遍历根目录下的所有子目录
for subdir, dirs, files in os.walk(root_dir):
    # 构造完整文件路径
    xml_file_path = os.path.join(subdir, file_template)
    
    # 获取当前子目录的名称
    subdir_name = os.path.basename(subdir)

    # 检查文件是否存在
    if os.path.isfile(xml_file_path):
        try:
            # 解析 XML 文件
            tree = ET.parse(xml_file_path)
            root = tree.getroot()
            
            # 提取资源数据
            resources = root.find('.//Resources')
            bram = resources.find('BRAM_18K').text
            dsp = resources.find('DSP').text
            ff = resources.find('FF').text
            lut = resources.find('LUT').text
            
            # 保存提取的信息
            performance_data.append(f"{subdir_name}: BRAM={bram}, DSP={dsp}, FF={ff}, LUT={lut}")

        except ET.ParseError as e:
            print(f"Error parsing {xml_file_path}: {e}")
        except Exception as e:
            print(f"Error reading data from {xml_file_path}: {e}")

# 将提取的信息写入到 txt 文件
output_file_path = os.path.join(root_dir, 'performance_estimates.txt')
with open(output_file_path, 'w') as output_file:
    for data in performance_data:
        output_file.write(data + '\n')

print(f"Performance data has been written to {output_file_path}.")



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
