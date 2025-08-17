"""Transformer blocks with KV cache, following Google Python Style."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionWithKVCache(nn.Module):
    """Multi-head self-attention with KV cache.

    Args:
        hidden_dim: Hidden size of the model.
        head_num: Number of attention heads.
    """

    def __init__(self, hidden_dim, head_num, use_rope=False, rope_base=10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num
        self.head_dim = hidden_dim // head_num
        self.use_rope = use_rope
        self.rope_base = rope_base
        
        assert hidden_dim % head_num == 0, "hidden_dim must be divisible by head_num"
        if self.use_rope:
            assert self.head_dim % 2 == 0, "head_dim must be even when using RoPE"
        
        # 线性变换层
        self.w_q = nn.Linear(hidden_dim, hidden_dim)
        self.w_k = nn.Linear(hidden_dim, hidden_dim)
        self.w_v = nn.Linear(hidden_dim, hidden_dim)
        self.w_o = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(self.head_dim)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates last-dim pairs (even-odd) as required by RoPE."""
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        x_rot = torch.stack((-x_odd, x_even), dim=-1)
        return x_rot.flatten(-2)
    
    def _compute_rope_cos_sin(self, seq_len: int, offset: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes cos/sin caches for rotary embedding.

        Returns tensors broadcastable to [batch, heads, seq_len, head_dim].
        """
        half_dim = self.head_dim // 2
        inv_freq = 1.0 / (self.rope_base ** (torch.arange(0, half_dim, device=device, dtype=torch.float32) / half_dim))
        t = torch.arange(offset, offset + seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum('s,d->sd', t, inv_freq)
        cos = torch.cos(freqs).unsqueeze(0).unsqueeze(0)
        sin = torch.sin(freqs).unsqueeze(0).unsqueeze(0)
        cos = torch.repeat_interleave(cos, repeats=2, dim=-1)
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)
        return cos, sin
    
    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        return (x * cos) + (self._rotate_half(x) * sin)
        
    def forward(self, x, kv_cache=None, use_cache=False):
        """Applies attention.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            kv_cache: Optional previous KV cache as a dict {'k': tensor, 'v': tensor}.
            use_cache: Whether to use and return the cache (generation mode).

        Returns:
            A tuple of (output, new_kv_cache) when use_cache is True, otherwise
            output only. Output has shape [batch_size, seq_len, hidden_dim].
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V
        Q = self.w_q(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.head_num, self.head_dim).transpose(1, 2)
        
        # RoPE on Q/K
        if self.use_rope:
            offset = 0
            if kv_cache is not None and use_cache and 'k' in kv_cache:
                offset = kv_cache['k'].size(2)
            cos, sin = self._compute_rope_cos_sin(seq_len, offset, x.device)
            Q = self._apply_rope(Q, cos, sin)
            K = self._apply_rope(K, cos, sin)
        
        # 如果有KV缓存，则拼接
        if kv_cache is not None and use_cache:
            if 'k' in kv_cache and 'v' in kv_cache:
                K = torch.cat([kv_cache['k'], K], dim=2)  # 在seq_len维度拼接
                V = torch.cat([kv_cache['v'], V], dim=2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Causal mask (GPT-style)
        if use_cache:
            # During generation: mask positions after the current one only.
            current_len = scores.size(-1)
            mask = torch.triu(
                torch.ones(seq_len, current_len), diagonal=current_len - seq_len + 1
            )
            mask = mask.unsqueeze(0).unsqueeze(0).to(scores.device)
            scores = scores.masked_fill(mask == 1, float('-inf'))
        else:
            # During training: standard causal mask.
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            mask = mask.unsqueeze(0).unsqueeze(0).to(scores.device)
            scores = scores.masked_fill(mask == 1, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        
        # 计算输出
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.w_o(out)
        
        # 准备新的KV缓存
        new_kv_cache = None
        if use_cache:
            new_kv_cache = {'k': K, 'v': V}
        
        return out, new_kv_cache

class TransformerLayer(nn.Module):
    """Single Transformer decoder block."""

    def __init__(self, hidden_dim, head_num, f_hidden_dim, use_rope=False, rope_base=10000.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_num = head_num

        self.self_attn = MultiHeadAttentionWithKVCache(hidden_dim, head_num, use_rope=use_rope, rope_base=rope_base)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, f_hidden_dim),
            nn.ReLU(),
            nn.Linear(f_hidden_dim, hidden_dim)
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, kv_cache=None, use_cache=False):
        """Runs attention and feed-forward with residual and layer norms.

        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim].
            kv_cache: Optional KV cache from previous steps.
            use_cache: Whether we are in generation mode.
        """
        # 自注意力机制 + 残差连接
        attn_out, new_kv_cache = self.self_attn(x, kv_cache, use_cache)
        x = self.norm1(x + attn_out)
        
        # 前馈网络 + 残差连接
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        return x, new_kv_cache

class Transformer(nn.Module):
    """GPT-style Transformer with positional embeddings and KV cache."""

    def __init__(self, vocab_size, hidden_dim, layer_num, head_num, max_len, f_hidden_dim=None, use_rope=False, rope_base=10000.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.head_num = head_num
        self.max_len = max_len
        self.use_rope = use_rope
        self.rope_base = rope_base
        
        if f_hidden_dim is None:
            f_hidden_dim = hidden_dim * 4  
            
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Embedding(max_len, hidden_dim)
        self.layers = nn.ModuleList([
            TransformerLayer(hidden_dim, head_num, f_hidden_dim, use_rope=use_rope, rope_base=rope_base)
            for _ in range(layer_num)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, kv_caches=None, use_cache=False):
        """Runs the Transformer.

        Args:
            x: Input token ids of shape [batch_size, seq_len].
            kv_caches: Optional list of KV caches per layer.
            use_cache: Whether to use and return KV caches.

        Returns:
            If use_cache is True, returns (logits, new_kv_caches). Otherwise
            returns logits only. Logits have shape [batch_size, seq_len, vocab_size].
        """
        batch_size, seq_len = x.shape
        
        # Token embedding + 位置编码（RoPE 时不使用可学习位置编码）
        if self.use_rope:
            x = self.embedding(x)
        else:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            if kv_caches is not None and use_cache:
                cache_len = kv_caches[0]['k'].size(2) if kv_caches[0] is not None else 0
                positions = positions + cache_len
            x = self.embedding(x) + self.pos_embedding(positions)
        
        # 通过每一层
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_kv_cache = layer(x, layer_kv_cache, use_cache)
            new_kv_caches.append(new_kv_cache)
        
        x = self.norm(x)
        logits = self.fc(x)
        
        if use_cache:
            return logits, new_kv_caches
        else:
            return logits
            
    def generate(self, input_ids, max_length=50, temperature=1.0, pad_token_id=0):
        """Greedy/temperature sampling generation using KV cache.

        Args:
            input_ids: Initial input ids [batch_size, initial_seq_len].
            max_length: Max length to generate.
            temperature: Sampling temperature.
            pad_token_id: Padding token id.
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # 初始化
        generated = input_ids.clone()
        kv_caches = [None] * self.layer_num
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # 只对最后一个token进行前向传播（利用KV cache）
                if kv_caches[0] is not None:
                    # 有缓存时，只需要处理最后一个token
                    last_token = generated[:, -1:] 
                    logits, kv_caches = self.forward(last_token, kv_caches, use_cache=True)
                else:
                    # 第一次，处理整个序列
                    logits, kv_caches = self.forward(generated, kv_caches, use_cache=True)
                
                # 只取最后一个位置的logits
                next_token_logits = logits[:, -1, :] / temperature
                
                # 采样下一个token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # 添加到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # 如果所有序列都生成了结束符，提前停止
                if (next_token == pad_token_id).all():
                    break
        
        return generated
