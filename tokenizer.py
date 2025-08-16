"""Simple character-level tokenizer for the arithmetic vocabulary."""

from typing import List

import torch
from config import Config


class Tokenizer:
    """Maps between characters and token ids using the global Config."""

    def __init__(self):
        self.config = Config()
        self.tokens = self.config.tokens
        self.id_to_token = self.config.id_to_token
        self.vocab_size = self.config.vocab_size
        self.pad_token_id = self.config.pad_token_id
        self.sop_token_id = self.config.sop_token_id
        self.eop_token_id = self.config.eop_token_id

    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token ids."""
        return [self.tokens[token] for token in text]

    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token ids back into a string."""
        return "".join([self.id_to_token[id] for id in ids])

