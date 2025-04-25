# codanet demo 介绍

#### 环境配置文件：environmen.yml

#### 不结合硬件反馈的NAS搜索代码：nas_vgg_no_hw.py （以VGGNet模型结构为例）

#### 结合硬件反馈的NAS搜索代码：nas_vgg_hw.py （以VGGNet模型结构为例）

#### 对训练的模型进行逻辑综合代码：
#### 1.单个模型：logic_single.py
#### 2.批量模型：for_logic.py

#### 生成比特流文件代码：bit_files.py

#### 从所有综合好的工程文件里提取资源预测模型需要的训练数据：for_read.py

#### 训练资源预测模型（xgboost）代码：xgboost_nihe.py
