"""Training and evaluation utilities with Google Python Style formatting."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from config import config
from data import test_dataset, train_dataset
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
        train_ds = data.Dataset.from_list(self.train_dataset)
        test_ds = data.Dataset.from_list(self.test_dataset)

        train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.to(device)
        best_loss = float('inf')

        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0
            for batch in train_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)

                optimizer.zero_grad()
                output = self.model(input_ids)
                loss = criterion(output.view(-1, config.vocab_size), target_ids.view(-1))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

            if (epoch + 1) % save_interval == 0:
                self.model.eval()
                with torch.no_grad():
                    total_loss = 0.0
                    for batch in test_loader:
                        input_ids = batch['input_ids'].to(device)
                        target_ids = batch['target_ids'].to(device)

                        output = self.model(input_ids)
                        loss = criterion(
                            output.view(-1, config.vocab_size), target_ids.view(-1)
                        )
                        total_loss += loss.item()

                    avg_loss = total_loss / len(test_loader)
                    print(
                        f"Epoch {epoch + 1}/{num_epochs}, Test Loss: {avg_loss:.4f}"
                    )

                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(self.model.state_dict(), 'best_model.pth')

    def evaluate(self, batch_size, device):
        """Evaluates the model on the test dataset."""
        test_ds = data.Dataset.from_list(self.test_dataset)
        test_loader = data.DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        with torch.no_grad():
            total_loss = 0.0
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                target_ids = batch['target_ids'].to(device)

                output = self.model(input_ids)
                loss = criterion(output.view(-1, config.vocab_size), target_ids.view(-1))
                total_loss += loss.item()

            avg_loss = total_loss / len(test_loader)
            print(f"Test Loss: {avg_loss:.4f}")

    def generate(self, input_ids, max_length, temperature, pad_token_id):
        """Generates tokens with the model using KV cache."""
        self.model.eval()
        device = input_ids.device
        _ = input_ids.shape[0]  # batch_size (unused)

        generated = input_ids.clone()
        kv_caches = [None] * self.model.layer_num

        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
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
        config.vocab_size, config.d_model, config.n_layers, config.n_heads, config.max_seq_len
    )
    trainer = TransformerTrainer(model, train_dataset, test_dataset, config)
    trainer.train(
        config.batch_size, config.num_epochs, config.learning_rate, config.device, config.save_interval
    )
    trainer.evaluate(config.batch_size, config.device)
    trainer.generate(torch.tensor([[config.sop_token_id]]), 10, 0.7, config.pad_token_id)