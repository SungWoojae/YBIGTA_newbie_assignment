import torch
from torch import nn, Tensor, LongTensor
from torch.optim import Adam
from transformers import PreTrainedTokenizer
from typing import Literal, List
import random

class Word2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        window_size: int,
        method: Literal["cbow", "skipgram"]
    ) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, d_model)
        self.weight = nn.Linear(d_model, vocab_size, bias=False)
        self.window_size = window_size
        self.method = method

    def embeddings_weight(self) -> Tensor:
        return self.embeddings.weight.detach()

    def fit(
        self,
        corpus: List[str],
        tokenizer: PreTrainedTokenizer,
        lr: float,
        num_epochs: int
    ) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(self.parameters(), lr=lr)

        tokenized_corpus = [tokenizer.encode(sentence, add_special_tokens=False) for sentence in corpus]

        for epoch in range(num_epochs):
            total_loss = 0.0  # Ensure float type

            for sentence in tokenized_corpus:
                if len(sentence) < 2 * self.window_size + 1:
                    continue

                if self.method == "cbow":
                    loss = self._train_cbow(sentence, criterion, optimizer)
                elif self.method == "skipgram":
                    loss = self._train_skipgram(sentence, criterion, optimizer)

                total_loss += float(loss)  # Ensure float addition

            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

    def _train_cbow(self, sentence: List[int], criterion, optimizer) -> float:
        total_loss = 0.0
        for idx in range(self.window_size, len(sentence) - self.window_size):
            context = (
                sentence[idx - self.window_size : idx] +
                sentence[idx + 1 : idx + 1 + self.window_size]
            )
            target = sentence[idx]

            context_tensor = LongTensor(context).unsqueeze(0)
            target_tensor = LongTensor([target])

            optimizer.zero_grad()
            embedded = self.embeddings(context_tensor).mean(dim=1)
            logits = self.weight(embedded)
            loss = criterion(logits, target_tensor)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return float(total_loss)

    def _train_skipgram(self, sentence: List[int], criterion, optimizer) -> float:
        total_loss = 0.0
        for idx in range(self.window_size, len(sentence) - self.window_size):
            target = sentence[idx]
            context = (
                sentence[idx - self.window_size : idx] +
                sentence[idx + 1 : idx + 1 + self.window_size]
            )

            target_tensor = LongTensor([target]).unsqueeze(0)
            optimizer.zero_grad()
            embedded = self.embeddings(target_tensor)
            logits = self.weight(embedded).squeeze(1)

            loss = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)  # Ensure tensor type
            for word in context:
                word_tensor = LongTensor([word])
                loss = loss + criterion(logits, word_tensor)  # Maintain tensor operation

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return float(total_loss)
