# Transformer Warmup - 简化版目录结构

使用简单算术运算训练GPT架构的transformer模型。

## 目录结构

```
transformer-warmup/
├── README.md                    # 项目说明
├── requirements.txt             # 依赖包
├── config.py                    # 所有配置参数
├── data.py                      # 数据生成、tokenizer、dataset
├── model.py                     # transformer模型结构
├── train.py                     # 训练脚本
├── evaluate.py                  # 推理和评估脚本
├── utils.py                     # 工具函数（日志、保存等）
├── checkpoints/                 # 模型保存目录
└── logs/                        # 训练日志目录
```

## 核心模块说明

### 1. `config.py` - 统一配置
- 模型参数配置（维度、层数等）
- 训练参数配置（学习率、batch size等）
- 数据参数配置（数字范围、序列长度等）

### 2. `data.py` - 数据处理
- 算术表达式生成器
- Tokenizer（词表：0-9, +, =, <pad>, <sop>, <eop>）
- PyTorch Dataset类

### 3. `model.py` - 模型结构
- GPT-style Transformer模型
- 多头注意力、前馈网络、位置编码等组件

### 4. `train.py` - 训练
- 训练循环、损失计算
- 验证和checkpointing
- 直接运行开始训练

### 5. `evaluate.py` - 推理评估
- 模型推理
- 准确率计算和错误分析
- 结果可视化

### 6. `utils.py` - 工具函数
- 日志记录
- 模型保存/加载
- 其他通用工具

## 使用方式

```bash
# 训练模型
python train.py

# 评估模型
python evaluate.py --checkpoint checkpoints/best_model.pt
```

这个结构更加简洁，将相关功能整合在一起，便于快速开发和理解。