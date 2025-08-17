"""Project-wide configuration in a single module.

Follows Google Python Style for docstrings and naming.
"""

import torch


class Config:
    """Unified configuration container for data, model and training settings."""

    def __init__(self):
        # ===== 词表和特殊token =====
        self.tokens = {
            '0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
            '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
            '+': 10, '=': 11,
            '<pad>': 12, '<sop>': 13, '<eop>': 14
        }
        self.id_to_token = {v: k for k, v in self.tokens.items()}
        self.vocab_size = len(self.tokens)
        self.pad_token_id = 12
        self.sop_token_id = 13
        self.eop_token_id = 14
        
        # ===== 数据生成参数 =====
        self.min_number = 0
        self.max_number = 999
        self.max_seq_len = 16
        self.train_size = 1024000
        self.val_size = 1024000
        self.test_size = 10000
        
        # ===== 模型参数 =====
        self.d_model = 128
        self.n_heads = 8
        self.n_layers = 10
        self.d_ff = 4 * self.d_model
        self.dropout = 0
        self.use_rope = True
        self.rope_base = 10000.0
        
        # ===== 训练参数 =====
        self.batch_size = 1024
        self.learning_rate = 2e-4
        self.num_epochs = 50
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_interval = 5  # 每5个epoch保存一次
        self.max_grad_norm = 0.5  # 梯度裁剪阈值，抑制梯度/损失 spike
        self.label_smoothing = 0.0  # 交叉熵标签平滑，减少尖锐预测的影响
        self.skip_non_finite = True  # 遇到非有限损失时跳过该步更新
        self.weight_decay = 0.03  # AdamW权重衰减，抑制参数范数增长
        self.max_param_norm = None  # 可选：参数范数上限（L2），如需硬性约束可设为数值，例如 10.0
        # 学习率调度
        self.scheduler_type = 'cosine'  # 'cosine' 或 'linear'
        self.warmup_ratio = 0.03  # 预热步数占总训练步数比例（0~1）
        self.min_lr_ratio = 0.1  # 最小学习率占峰值学习率的比例（0~1）
        
        # ===== 其他 =====
        self.seed = 42

# Global configuration instance.
config = Config()
