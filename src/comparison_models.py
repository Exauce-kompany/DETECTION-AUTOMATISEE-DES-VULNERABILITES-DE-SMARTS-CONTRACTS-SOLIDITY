"""Matched small neural models for the architecture and context-length study."""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def masked_pool(values, mask):
    mask = mask.unsqueeze(-1)
    mean = (values * mask).sum(1) / mask.sum(1).clamp_min(1)
    maximum = values.masked_fill(~mask, -1e9).amax(1)
    maximum = torch.where(mask.any(1), maximum, torch.zeros_like(maximum))
    return torch.cat((mean, maximum), dim=-1)


class ContractClassifier(nn.Module):
    def __init__(self, architecture, vocabulary_size, config):
        super().__init__()
        if architecture not in ("cnn", "bilstm", "cnn_bilstm"):
            raise ValueError(architecture)
        self.architecture = architecture
        self.window_size = config["window_size"]
        dim, filters, units = config["embedding_dim"], config["convolution_filters"], config["lstm_units"]
        self.embedding = nn.Embedding(vocabulary_size, dim, padding_idx=0)
        if architecture in ("cnn", "cnn_bilstm"):
            self.conv = nn.Conv1d(dim, filters, config["convolution_kernel"], padding="same")
        if architecture == "bilstm":
            # Two fused CPU LSTMs avoid the slow per-step packed-sequence path.
            # Reverse only actual tokens, never padding, so both directions are
            # equivalent to a length-aware bidirectional recurrent network.
            self.forward_lstm = nn.LSTM(dim, units, batch_first=True)
            self.backward_lstm = nn.LSTM(dim, units, batch_first=True)
            features = units * 4
        elif architecture == "cnn_bilstm":
            self.recurrent = nn.LSTM(filters * 2, units, batch_first=True, bidirectional=True)
            features = units * 4
        else:
            features = filters * 4
        self.head = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(features, config["dense_units"]), nn.ReLU(), nn.Linear(config["dense_units"], 1))

    def forward(self, ids, lengths):
        if self.architecture == "bilstm":
            # A true token-level BiLSTM, preserving all dependencies in the input.
            embedded = self.embedding(ids)
            positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
            reverse = (lengths[:, None] - 1 - positions).clamp_min(0)
            reverse = reverse.unsqueeze(-1)
            forward, _ = self.forward_lstm(embedded)
            backward, _ = self.backward_lstm(embedded.gather(1, reverse.expand_as(embedded)))
            values = torch.cat((forward, backward.gather(1, reverse.expand_as(backward))), dim=-1)
            pooled = masked_pool(values, ids.ne(0))
        else:
            batch, total = ids.shape
            if total % self.window_size:
                ids = nn.functional.pad(ids, (0, self.window_size - total % self.window_size))
            windows = ids.reshape(batch, -1, self.window_size)
            flat = windows.flatten(0, 1)
            features = torch.relu(self.conv(self.embedding(flat).transpose(1, 2))).transpose(1, 2)
            features = masked_pool(features, flat.ne(0)).reshape(batch, windows.shape[1], -1)
            window_mask = windows.ne(0).any(-1)
            if self.architecture == "cnn_bilstm":
                packed = pack_padded_sequence(features, window_mask.sum(1).cpu(), batch_first=True, enforce_sorted=False)
                outputs, _ = self.recurrent(packed)
                features, _ = pad_packed_sequence(outputs, batch_first=True, total_length=windows.shape[1])
            pooled = masked_pool(features, window_mask)
        return self.head(pooled).squeeze(-1)


class FeatureClassifier(nn.Module):
    def __init__(self, input_dim, config):
        super().__init__()
        self.network = nn.Sequential(nn.Dropout(config["dropout"]), nn.Linear(input_dim, config["dense_units"]), nn.ReLU(), nn.Linear(config["dense_units"], 1))

    def forward(self, values):
        return self.network(values).squeeze(-1)
