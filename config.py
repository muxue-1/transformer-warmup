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
        self.max_number = 9999
        self.max_seq_len = 32
        self.train_size = 50000
        self.val_size = 5000
        self.test_size = 5000
        
        # ===== 模型参数 =====
        self.d_model = 128
        self.n_heads = 8
        self.n_layers = 6
        self.d_ff = 512
        self.dropout = 0.1
        
        # ===== 训练参数 =====
        self.batch_size = 64
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.save_interval = 5  # 每5个epoch保存一次
        
        # ===== 其他 =====
        self.seed = 42

# Global configuration instance.
config = Config()
