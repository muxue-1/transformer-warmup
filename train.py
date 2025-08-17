"""Training and evaluation utilities with Google Python Style formatting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
try:
    from tqdm import tqdm
except Exception:  # Fallback when tqdm is not installed
    def tqdm(iterable=None, total=None, desc=None):
        return iterable if iterable is not None else range(total or 0)

from config import config
from data import test_dataset, val_dataset, train_dataset, EquationDataset
from model import Transformer


class TransformerTrainer:
    """Simple trainer for the Transformer model.

    Attributes:
        model: The Transformer model to train.
        train_dataset: Training data as a list of dicts.
        test_dataset: Test data as a list of dicts.
        config: Global configuration.
    """

    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

    def train(self, batch_size, num_epochs, learning_rate, device, save_interval):
        """Runs the training loop with periodic evaluation and checkpointing."""
        train_ds = EquationDataset(self.train_dataset, self.config)
        test_ds = EquationDataset(self.test_dataset, self.config)

        train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        # Use AdamW with decoupled weight decay and proper param groups (no decay on bias/LayerNorm)
        weight_decay = getattr(self.config, 'weight_decay', 0.0)
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.dim() == 1 or name.endswith('.bias'):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        optimizer = optim.AdamW(param_groups, lr=learning_rate)

        # Build LR scheduler with warmup + decay per step
        total_steps = max(1, (len(train_loader) * num_epochs))
        warmup_steps = int(getattr(self.config, 'warmup_ratio', 0.0) * total_steps)
        min_lr_ratio = float(getattr(self.config, 'min_lr_ratio', 0.0))
        scheduler_type = getattr(self.config, 'scheduler_type', 'cosine')

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps and warmup_steps > 0:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            progress = min(max(progress, 0.0), 1.0)
            if scheduler_type == 'linear':
                return (1.0 - progress) * (1.0 - min_lr_ratio) + min_lr_ratio
            else:  # cosine
                import math
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                return cosine_decay * (1.0 - min_lr_ratio) + min_lr_ratio

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        criterion = nn.CrossEntropyLoss(
            label_smoothing=getattr(self.config, 'label_smoothing', 0.0),
            ignore_index=config.pad_token_id,
        )

        self.model.to(device)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            correct_tokens_train = 0
            total_tokens_train = 0
            pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Train")
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)

                optimizer.zero_grad()
                output = self.model(input_ids)
                loss = criterion(output.view(-1, config.vocab_size), target_ids.view(-1))
                # Train accuracy (ignore pad)
                preds_train = output.argmax(dim=-1)
                mask_train = target_ids.ne(config.pad_token_id)
                correct_tokens_train += (preds_train.eq(target_ids) & mask_train).sum().item()
                total_tokens_train += mask_train.sum().item()
                # Retain grad on loss to inspect d(loss)/d(loss) (typically 1.0 after backward)
                try:
                    loss.retain_grad()
                except Exception:
                    pass
                # Check non-finite loss
                if getattr(self.config, 'skip_non_finite', True) and not torch.isfinite(loss):
                    continue

                loss.backward()
                # Gradient clipping
                total_grad_norm = None
                if getattr(self.config, 'max_grad_norm', None):
                    total_grad_norm = nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    total_grad_norm = float(total_grad_norm)
                else:
                    grads = [
                        p.grad.detach().data.norm(2)
                        for p in self.model.parameters()
                        if p.grad is not None
                    ]
                    if len(grads) > 0:
                        total_grad_norm = float(torch.norm(torch.stack(grads), 2))
                    else:
                        total_grad_norm = 0.0


                # Optional detailed gradient prints
                if getattr(self.config, 'debug_print_grads', False):
                    # Print loss grad (usually 1.0)
                    try:
                        pbar.write(f"loss.grad: {loss.grad.item() if loss.grad is not None else 'None'}")
                    except Exception:
                        pass
                    # Print top-K parameter grad norms
                    try:
                        top_k = int(getattr(self.config, 'grad_print_topk', 3))
                        named_grads = []
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                named_grads.append((name, float(param.grad.data.norm(2))))
                        named_grads.sort(key=lambda x: x[1], reverse=True)
                        for name, gnorm in named_grads[:top_k]:
                            pbar.write(f"grad ||{name}||_2 = {gnorm:.6f}")
                    except Exception:
                        pass

                optimizer.step()
                scheduler.step()

                # Optional hard clamp on parameter norms to keep model_norm bounded
                if getattr(self.config, 'max_param_norm', None):
                    with torch.no_grad():
                        current_norms = [p.data.norm(2) for p in self.model.parameters() if p is not None]
                        current_total_norm = torch.norm(torch.stack(current_norms), 2).item() if len(current_norms) > 0 else 0.0
                        max_norm = float(self.config.max_param_norm)
                        if current_total_norm > max_norm and current_total_norm > 0:
                            scale = max_norm / current_total_norm
                            for p in self.model.parameters():
                                if p is not None:
                                    p.data.mul_(scale)

                total_loss += loss.item()
                try:
                    current_lr = scheduler.get_last_lr()[0]
                except Exception:
                    current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(
                    avg_loss=f"{total_loss / max(1, pbar.n):.4f}",
                    grad_norm=f"{total_grad_norm:.4f}",
                    acc=f"{(correct_tokens_train / max(1, total_tokens_train)):.4f}",
                    lr=f"{current_lr:.2e}"
                )

            avg_loss = total_loss / len(train_loader)
            train_acc = correct_tokens_train / max(1, total_tokens_train)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

            if (epoch + 1) % save_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    total_loss = 0.0
                    # 使用验证集进行周期性评估
                    val_ds = EquationDataset(val_dataset, self.config)
                    val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
                    pbar_eval = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch + 1}/{num_epochs} - Eval")
                    correct_tokens = 0
                    total_tokens = 0
                    for batch in pbar_eval:
                        input_ids = batch['input_ids'].to(device)
                        target_ids = batch['target_ids'].to(device)

                        output = self.model(input_ids)
                        loss = criterion(
                            output.view(-1, config.vocab_size), target_ids.view(-1)
                        )
                        total_loss += loss.item()
                        # accuracy（忽略 pad）
                        preds = output.argmax(dim=-1)
                        mask = target_ids.ne(config.pad_token_id)
                        correct_tokens += (preds.eq(target_ids) & mask).sum().item()
                        total_tokens += mask.sum().item()
                        acc = correct_tokens / max(1, total_tokens)
                        pbar_eval.set_postfix(avg_loss=f"{total_loss / max(1, pbar_eval.n):.4f}", acc=f"{acc:.4f}")

                    avg_loss = total_loss / len(val_loader)
                    val_acc = correct_tokens / max(1, total_tokens)
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(self.model.state_dict(), 'best_model.pth')

    def evaluate(self, batch_size, device):
        """Evaluates the model on the test dataset."""
        test_ds = EquationDataset(self.test_dataset, self.config)
        test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss(ignore_index=config.pad_token_id)
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            pbar = tqdm(test_loader, total=len(test_loader), desc="Evaluate")
            correct_tokens = 0
            total_tokens = 0
            for batch in pbar:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)

                output = self.model(input_ids)
                loss = criterion(output.view(-1, config.vocab_size), target_ids.view(-1))
                total_loss += loss.item()
                preds = output.argmax(dim=-1)
                mask = target_ids.ne(config.pad_token_id)
                correct_tokens += (preds.eq(target_ids) & mask).sum().item()
                total_tokens += mask.sum().item()
                acc = correct_tokens / max(1, total_tokens)
                pbar.set_postfix(avg_loss=f"{total_loss / max(1, pbar.n):.4f}", acc=f"{acc:.4f}")

            avg_loss = total_loss / len(test_loader)
            test_acc = correct_tokens / max(1, total_tokens)
            print(f"Test Loss: {avg_loss:.4f}, Test Acc: {test_acc:.4f}")

    def generate(self, input_ids, max_length, temperature, pad_token_id):
        """Generates tokens with the model using KV cache."""
        self.model.eval()
        device = input_ids.device
        _ = input_ids.shape[0]  # batch_size (unused)

        generated = input_ids.clone()
        kv_caches = [None] * self.model.layer_num

        with torch.no_grad():
            for _ in tqdm(range(max_length - input_ids.shape[1]), total=max(0, max_length - input_ids.shape[1]), desc="Generate"):
                if kv_caches[0] is not None:
                    last_token = generated[:, -1:]
                    logits, kv_caches = self.model.forward(
                        last_token, kv_caches, use_cache=True
                    )
                else:
                    logits, kv_caches = self.model.forward(
                        generated, kv_caches, use_cache=True
                    )

                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                generated = torch.cat([generated, next_token], dim=1)

                if (next_token == pad_token_id).all():
                    break

        return generated


if __name__ == "__main__":
    model = Transformer(
        config.vocab_size,
        config.d_model,
        config.n_layers,
        config.n_heads,
        config.max_seq_len,
        use_rope=config.use_rope,
        rope_base=config.rope_base,
    )
    trainer = TransformerTrainer(model, train_dataset, test_dataset, config)
    trainer.train(
        config.batch_size, config.num_epochs, config.learning_rate, config.device, config.save_interval
    )
    trainer.evaluate(config.batch_size, config.device)
    # trainer.generate(torch.tensor([[config.sop_token_id]]), 10, 0.7, config.pad_token_id)